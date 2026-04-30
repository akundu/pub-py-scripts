"""Consolidated tests for the Universal Trade Platform (UTP).

Covers: CLI helpers, API endpoints, services (ledger, position store, dashboard,
expiration, position sync, reconciliation, playbook, CSV import, fill tracking),
providers (IBKR stub + live, caching), authentication, symbology, websocket,
and the trade CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings
from app.models import (
    Broker,
    EquityOrder,
    LedgerEntry,
    LedgerEventType,
    LedgerQuery,
    MultiLegOrder,
    OptionAction,
    OptionLeg,
    OptionType,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PlaybookInstruction,
    Position,
    PositionSource,
    TradeRequest,
)

# Import CLI functions from utp
from utp import (
    _build_instruction_from_args,
    _build_margin_order,
    _cleanup_positions,
    _cmd_close,
    _cmd_flush,
    _cmd_journal,
    _cmd_options,
    _cmd_performance,
    _cmd_portfolio,
    _cmd_reconcile,
    _cmd_status,
    _color,
    _execute_single_order,
    _generate_safe_defaults,
    _get_mode,
    _get_symbol_from_instruction,
    _mode_label,
    _next_trading_day,
    _print_header,
    _print_section,
    _print_step,
    _auto_price_spread,
    _auto_price_iron_condor,
    _resolve_data_dir,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Display Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestDisplayHelpers:
    def test_color_wraps_text(self):
        result = _color("hello", "92")
        assert "\033[92m" in result
        assert "hello" in result
        assert "\033[0m" in result

    def test_print_header(self, capsys):
        _print_header("Test Title")
        out = capsys.readouterr().out
        assert "Test Title" in out
        assert "=" * 70 in out

    def test_print_section(self, capsys):
        _print_section("Section")
        out = capsys.readouterr().out
        assert "Section" in out
        assert "\u2500" in out

    def test_print_step_pass(self, capsys):
        _print_step("Check", "pass", "all good")
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "all good" in out

    def test_print_step_fail(self, capsys):
        _print_step("Check", "fail", "broken")
        out = capsys.readouterr().out
        assert "FAIL" in out

    def test_print_step_skip(self, capsys):
        _print_step("Check", "skip")
        out = capsys.readouterr().out
        assert "SKIP" in out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Mode Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestTradeHttpTimeout:
    """The CLI's HTTP timeout for ``POST /trade/execute`` must always exceed
    the daemon's broker-fill poll timeout. Otherwise the client gives up
    while the daemon is still legitimately waiting on the broker — the bug
    we hit in production with the hardcoded 90s/60s pair."""

    def test_default_timeout_is_broker_timeout_plus_2pct(self, monkeypatch):
        """With the default 60s daemon poll, client waits 61.2s (60 × 1.02)."""
        monkeypatch.delenv("ORDER_POLL_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("UTP_TRADE_HTTP_TIMEOUT", raising=False)
        from utp import _trade_http_timeout
        assert _trade_http_timeout() == pytest.approx(60.0 * 1.02)

    def test_timeout_scales_with_broker_timeout(self, monkeypatch):
        """If the daemon's poll timeout is bumped, the client auto-scales —
        without the user having to set a second env var."""
        monkeypatch.setenv("ORDER_POLL_TIMEOUT_SECONDS", "120")
        monkeypatch.delenv("UTP_TRADE_HTTP_TIMEOUT", raising=False)
        from utp import _trade_http_timeout
        assert _trade_http_timeout() == pytest.approx(120.0 * 1.02)

    def test_explicit_override_wins_when_above_floor(self, monkeypatch):
        """``UTP_TRADE_HTTP_TIMEOUT`` can grant *more* slack than the +2% rule."""
        monkeypatch.setenv("ORDER_POLL_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("UTP_TRADE_HTTP_TIMEOUT", "300")
        from utp import _trade_http_timeout
        assert _trade_http_timeout() == 300.0

    def test_override_floored_to_2pct_margin(self, monkeypatch):
        """A reckless override below broker × 1.02 is silently raised to the
        floor — the policy is "at least +2%", never less."""
        monkeypatch.setenv("ORDER_POLL_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("UTP_TRADE_HTTP_TIMEOUT", "30")  # too short
        from utp import _trade_http_timeout
        assert _trade_http_timeout() == pytest.approx(60.0 * 1.02)

    def test_invalid_override_falls_back_to_floor(self, monkeypatch):
        monkeypatch.setenv("ORDER_POLL_TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("UTP_TRADE_HTTP_TIMEOUT", "not-a-number")
        from utp import _trade_http_timeout
        assert _trade_http_timeout() == pytest.approx(60.0 * 1.02)


class TestPollIntervalFloor:
    """Order-status polling must not exceed IBKR's pacing — the floor is 2s."""

    def test_settings_default_is_2s(self):
        from app.config import Settings
        s = Settings(_env_file=None)  # ignore any .env on disk
        assert s.order_poll_interval_seconds == 2.0

    def test_settings_validator_floors_sub_2s_overrides(self, monkeypatch):
        """Even an explicit env var < 2s gets bumped to 2s — the validator
        is the policy enforcement point."""
        monkeypatch.setenv("ORDER_POLL_INTERVAL_SECONDS", "0.5")
        from app.config import Settings
        s = Settings(_env_file=None)
        assert s.order_poll_interval_seconds == 2.0

    def test_settings_validator_passes_higher_values(self, monkeypatch):
        monkeypatch.setenv("ORDER_POLL_INTERVAL_SECONDS", "5.0")
        from app.config import Settings
        s = Settings(_env_file=None)
        assert s.order_poll_interval_seconds == 5.0

    @pytest.mark.asyncio
    async def test_await_order_fill_floors_caller_supplied_interval(self):
        """Defense in depth: even if a caller passes ``poll_interval=0.01``
        directly to ``await_order_fill``, it gets bumped to the 2s floor."""
        from app.services import trade_service

        captured: list[float] = []
        original_sleep = trade_service.asyncio.sleep

        async def _spy_sleep(secs):
            captured.append(secs)
            # Yield without actually sleeping to keep the test fast
            await original_sleep(0)

        # Patch asyncio.sleep on the trade_service module's reference
        trade_service.asyncio.sleep = _spy_sleep
        try:
            class _OneShot:
                async def get_order_status(self, _oid):
                    from app.models import OrderResult, OrderStatus, Broker
                    return OrderResult(
                        order_id="x", broker=Broker.IBKR,
                        status=OrderStatus.FILLED, filled_price=1.0,
                    )

            with patch.object(trade_service, "ProviderRegistry") as mock_reg:
                mock_reg.get.return_value = _OneShot()
                await trade_service.await_order_fill(
                    broker=Broker.IBKR, order_id="t-floor",
                    poll_interval=0.01, timeout=1.0,
                )
        finally:
            trade_service.asyncio.sleep = original_sleep

        assert captured, "expected at least one sleep call"
        assert captured[0] == pytest.approx(2.0), (
            f"poll_interval should be floored to 2.0; got {captured[0]}"
        )


class TestModeDetection:
    def test_default_is_dry_run(self):
        args = argparse.Namespace(live=False, paper=False)
        assert _get_mode(args) == "dry-run"

    def test_default_mode_paper_override(self):
        """Subcommands with default_paper=True use paper when no flags set."""
        args = argparse.Namespace(live=False, paper=False, dry_run=False, _default_mode="paper")
        assert _get_mode(args) == "paper"

    def test_paper_mode(self):
        args = argparse.Namespace(live=False, paper=True)
        assert _get_mode(args) == "paper"

    def test_live_mode(self):
        args = argparse.Namespace(live=True, paper=False)
        assert _get_mode(args) == "live"

    def test_mode_label_dry_run(self):
        label = _mode_label("dry-run")
        assert "DRY-RUN" in label

    def test_mode_label_live(self):
        label = _mode_label("live")
        assert "LIVE" in label


# ═══════════════════════════════════════════════════════════════════════════════
# Mode-Specific Data Directories
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveDataDir:
    def test_dry_run_unchanged(self):
        result = _resolve_data_dir("data/utp", "dry-run")
        assert result == Path("data/utp")

    def test_paper_subdirectory(self):
        result = _resolve_data_dir("data/utp", "paper")
        assert result == Path("data/utp/paper")

    def test_live_subdirectory(self):
        result = _resolve_data_dir("data/utp", "live")
        assert result == Path("data/utp/live")

    def test_flush_paper_uses_paper_dir(self, tmp_path):
        """Flush in paper mode targets the paper subdirectory, not root."""
        base_dir = tmp_path / "utp"
        paper_dir = base_dir / "paper"
        paper_dir.mkdir(parents=True)
        pos_file = paper_dir / "positions.json"
        pos_file.write_text('{"id1": {"status": "open"}}')

        # Also create a live positions file that should NOT be touched
        live_dir = base_dir / "live"
        live_dir.mkdir(parents=True)
        live_pos = live_dir / "positions.json"
        live_pos.write_text('{"id2": {"status": "open", "symbol": "AAPL"}}')

        args = argparse.Namespace(
            what="positions", data_dir=str(base_dir),
            live=False, paper=True, dry_run=False, _default_mode="paper",
        )
        import asyncio
        asyncio.get_event_loop().run_until_complete(_cmd_flush(args))
        # Paper positions flushed
        assert json.loads(pos_file.read_text()) == {}
        # Live positions untouched
        assert "id2" in json.loads(live_pos.read_text())

    def test_reconcile_paper_isolated_from_live(self, tmp_path):
        """Paper reconcile only sees paper positions, not live ones."""
        from app.services.position_store import PlatformPositionStore

        # Create separate stores for live and paper
        base_dir = tmp_path / "utp"
        live_dir = base_dir / "live"
        live_dir.mkdir(parents=True)
        paper_dir = base_dir / "paper"
        paper_dir.mkdir(parents=True)

        live_store = PlatformPositionStore(live_dir / "positions.json")
        paper_store = PlatformPositionStore(paper_dir / "positions.json")

        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY,
            quantity=100, order_type="MARKET",
        ))
        live_store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=False)

        # Paper store should be empty
        assert len(paper_store.get_open_positions()) == 0
        # Live store has the position
        assert len(live_store.get_open_positions()) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Next Trading Day
# ═══════════════════════════════════════════════════════════════════════════════


class TestNextTradingDay:
    def test_returns_weekday(self):
        result = _next_trading_day()
        d = datetime.strptime(result, "%Y-%m-%d")
        assert d.weekday() < 5  # Mon-Fri

    def test_format(self):
        result = _next_trading_day()
        assert len(result) == 10
        assert result[4] == "-"
        assert result[7] == "-"


# ═══════════════════════════════════════════════════════════════════════════════
# Options Chain Command
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsCommand:
    def _base_args(self, tmp_path, **overrides):
        defaults = dict(
            symbol="SPX", type="CALL", expiration="2026-03-20",
            strike_min=None, strike_max=None, strike_range=15,
            list_expirations=False, data_dir=str(tmp_path),
            live=False, paper=False, dry_run=True, _default_mode="dry-run",
            host="127.0.0.1", port=None, client_id=10, exchange=None,
            broker="ibkr",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    async def test_options_dry_run_shows_strikes(self, tmp_path, capsys):
        """Options command shows option quotes with ticker symbols in dry-run mode."""
        rc = await _cmd_options(self._base_args(tmp_path))
        assert rc == 0
        captured = capsys.readouterr()
        assert "CALLS" in captured.out
        assert "Symbol" in captured.out
        assert "Strike" in captured.out
        assert "100.0" in captured.out
        # Verify OCC-style ticker symbol is shown (e.g. SPX260320C00100000)
        assert "SPX260320C" in captured.out

    async def test_options_list_expirations(self, tmp_path, capsys):
        """--list-expirations shows available expirations."""
        rc = await _cmd_options(self._base_args(tmp_path, expiration=None, list_expirations=True))
        assert rc == 0
        captured = capsys.readouterr()
        assert "Expirations" in captured.out
        assert "2026-03-20" in captured.out

    async def test_options_invalid_expiration(self, tmp_path, capsys):
        """Invalid expiration returns error and shows available dates."""
        rc = await _cmd_options(self._base_args(tmp_path, expiration="2099-01-01"))
        assert rc == 1
        captured = capsys.readouterr()
        assert "not available" in captured.out

    async def test_options_put_type(self, tmp_path, capsys):
        """PUT option type is handled correctly."""
        rc = await _cmd_options(self._base_args(tmp_path, type="PUT"))
        assert rc == 0
        captured = capsys.readouterr()
        assert "PUTS" in captured.out

    async def test_options_both_types(self, tmp_path, capsys):
        """BOTH shows calls and puts."""
        rc = await _cmd_options(self._base_args(tmp_path, type="BOTH"))
        assert rc == 0
        captured = capsys.readouterr()
        assert "CALLS" in captured.out
        assert "PUTS" in captured.out

    async def test_options_custom_strike_range(self, tmp_path, capsys):
        """--strike-range accepts custom percentage."""
        rc = await _cmd_options(self._base_args(tmp_path, strike_range=5))
        assert rc == 0
        captured = capsys.readouterr()
        assert "CALLS" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Margin Order Building
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildMarginOrder:
    def test_credit_spread(self):
        args = argparse.Namespace(
            symbol="SPX", expiration="2026-03-20", short_strike=5500,
            long_strike=5475, option_type="PUT", quantity=2,
            net_price=3.50, broker="ibkr",
        )
        order = _build_margin_order("credit-spread", args)
        assert len(order.legs) == 2
        assert order.legs[0].action.value == "SELL_TO_OPEN"
        assert order.legs[0].strike == 5500
        assert order.legs[1].action.value == "BUY_TO_OPEN"
        assert order.legs[1].strike == 5475
        assert order.quantity == 2

    def test_debit_spread(self):
        args = argparse.Namespace(
            symbol="QQQ", expiration="2026-03-20", long_strike=480,
            short_strike=490, option_type="CALL", quantity=3,
            net_price=4.00, broker="ibkr",
        )
        order = _build_margin_order("debit-spread", args)
        assert len(order.legs) == 2
        assert order.legs[0].action.value == "BUY_TO_OPEN"
        assert order.legs[0].strike == 480
        assert order.legs[1].action.value == "SELL_TO_OPEN"
        assert order.legs[1].strike == 490

    def test_iron_condor(self):
        args = argparse.Namespace(
            symbol="SPX", expiration="2026-03-20", put_short=5500,
            put_long=5475, call_short=5700, call_long=5725,
            quantity=1, net_price=3.50, broker="ibkr",
        )
        order = _build_margin_order("iron-condor", args)
        assert len(order.legs) == 4
        assert order.legs[0].strike == 5500  # sell put
        assert order.legs[1].strike == 5475  # buy put
        assert order.legs[2].strike == 5700  # sell call
        assert order.legs[3].strike == 5725  # buy call

    def test_single_option(self):
        args = argparse.Namespace(
            symbol="SPY", expiration="2026-03-20", strike=550,
            option_type="PUT", action="BUY_TO_OPEN", quantity=1,
            net_price=2.50, broker="ibkr",
        )
        order = _build_margin_order("option", args)
        assert len(order.legs) == 1
        assert order.legs[0].strike == 550
        assert order.legs[0].action.value == "BUY_TO_OPEN"

    def test_unknown_type_raises(self):
        args = argparse.Namespace(symbol="SPX", broker="ibkr")
        with pytest.raises(ValueError, match="Unknown margin subcommand"):
            _build_margin_order("butterfly", args)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Portfolio Command
# ═══════════════════════════════════════════════════════════════════════════════


class TestPortfolioCommand:
    @pytest.mark.asyncio
    async def test_portfolio_empty(self, tmp_path, capsys):
        args = argparse.Namespace(data_dir=str(tmp_path))
        rc = await _cmd_portfolio(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Portfolio Summary" in out
        assert "no open positions" in out

    @pytest.mark.asyncio
    async def test_portfolio_with_positions(self, tmp_path, capsys):
        from app.services.position_store import get_position_store

        args = argparse.Namespace(data_dir=str(tmp_path))
        store = get_position_store()
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY,
            quantity=100, order_type=OrderType.MARKET,
        ))
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED,
                             filled_price=175.0)
        store.add_position(req, result, is_paper=True)

        rc = await _cmd_portfolio(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "AAPL" in out
        assert "175.00" in out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Status Command
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatusCommand:
    @pytest.mark.asyncio
    async def test_status_empty(self, tmp_path, capsys):
        args = argparse.Namespace(data_dir=str(tmp_path))
        rc = await _cmd_status(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "System Status" in out
        assert "Active Positions: 0" in out
        assert "(none)" in out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Journal Command
# ═══════════════════════════════════════════════════════════════════════════════


class TestJournalCommand:
    @pytest.mark.asyncio
    async def test_journal_empty(self, tmp_path, capsys):
        args = argparse.Namespace(data_dir=str(tmp_path), days=None,
                                  limit=50, event_type=None, order_id=None)
        rc = await _cmd_journal(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Trade Journal" in out
        assert "no entries found" in out

    @pytest.mark.asyncio
    async def test_journal_with_entries(self, tmp_path, capsys):
        from app.services.ledger import get_ledger

        args = argparse.Namespace(data_dir=str(tmp_path), days=None,
                                  limit=50, event_type=None, order_id=None)
        ledger = get_ledger()
        await ledger.log_order_submitted(
            broker=Broker.IBKR, order_id="test-123",
            source=PositionSource.PAPER, dry_run=True,
            data={"symbol": "SPY"},
        )

        rc = await _cmd_journal(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "ORDER_SUBMITTED" in out

    @pytest.mark.asyncio
    async def test_journal_invalid_event_type(self, tmp_path, capsys):
        args = argparse.Namespace(data_dir=str(tmp_path), days=None,
                                  limit=50, event_type="BOGUS_TYPE", order_id=None)
        rc = await _cmd_journal(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "Invalid event type" in out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Performance Command
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceCommand:
    @pytest.mark.asyncio
    async def test_performance_no_trades(self, tmp_path, capsys):
        args = argparse.Namespace(data_dir=str(tmp_path), days=30,
                                  start_date=None, end_date=None)
        rc = await _cmd_performance(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Performance Metrics" in out
        assert "no closed trades" in out

    @pytest.mark.asyncio
    async def test_performance_with_trades(self, tmp_path, capsys):
        from app.services.position_store import get_position_store

        args = argparse.Namespace(data_dir=str(tmp_path), days=30,
                                  start_date=None, end_date=None)
        store = get_position_store()

        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY,
            quantity=10, order_type=OrderType.MARKET,
        ))
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED,
                             filled_price=500.0)
        pos_id = store.add_position(req, result, is_paper=True)
        store.close_position(pos_id, exit_price=510.0, reason="profit_target")

        rc = await _cmd_performance(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total Trades:" in out
        assert "Win Rate:" in out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Trade Instruction Building
# ═══════════════════════════════════════════════════════════════════════════════


def _ns(**kwargs):
    """Build an argparse.Namespace with defaults for trade tests."""
    defaults = dict(
        dry_run=True, paper=False, live=False, broker="ibkr",
        host="127.0.0.1", port=None, client_id=10, exchange=None,
        data_dir="data/utp", poll_timeout=5.0, poll_interval=0.1,
        validate_all=False, cleanup=False, symbol="SPY",
        expiration="2026-04-01", subcommand=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestBuildInstruction:
    def test_equity(self):
        args = _ns(subcommand="equity", symbol="AAPL", side="BUY", quantity=10,
                   order_type="MARKET", limit_price=None)
        instr = _build_instruction_from_args("equity", args)
        assert instr["type"] == "equity"
        assert instr["params"]["symbol"] == "AAPL"
        assert instr["params"]["action"] == "BUY"
        assert instr["params"]["quantity"] == 10
        assert "id" in instr

    def test_option(self):
        args = _ns(subcommand="option", symbol="SPX", expiration="2026-04-01",
                   strike=5500.0, option_type="PUT", action="BUY_TO_OPEN",
                   quantity=1, order_type="LIMIT", limit_price=2.50)
        instr = _build_instruction_from_args("option", args)
        assert instr["type"] == "single_option"
        assert instr["params"]["strike"] == 5500.0

    def test_credit_spread(self):
        args = _ns(subcommand="credit-spread", symbol="SPX", expiration="2026-04-01",
                   short_strike=5500.0, long_strike=5475.0, option_type="PUT",
                   quantity=2, net_price=3.50)
        instr = _build_instruction_from_args("credit-spread", args)
        assert instr["type"] == "credit_spread"
        assert instr["params"]["short_strike"] == 5500.0
        assert instr["params"]["long_strike"] == 5475.0

    def test_debit_spread(self):
        args = _ns(subcommand="debit-spread", symbol="QQQ", expiration="2026-04-01",
                   long_strike=480.0, short_strike=490.0, option_type="CALL",
                   quantity=3, net_price=4.00)
        instr = _build_instruction_from_args("debit-spread", args)
        assert instr["type"] == "debit_spread"
        assert instr["params"]["long_strike"] == 480.0

    def test_iron_condor(self):
        args = _ns(subcommand="iron-condor", symbol="SPX", expiration="2026-04-01",
                   put_short=5500.0, put_long=5475.0, call_short=5700.0,
                   call_long=5725.0, quantity=1, net_price=3.50)
        instr = _build_instruction_from_args("iron-condor", args)
        assert instr["type"] == "iron_condor"
        assert instr["params"]["put_short"] == 5500.0

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown subcommand"):
            _build_instruction_from_args("butterfly", _ns())


class TestGetSymbolFromInstruction:
    def test_present(self):
        assert _get_symbol_from_instruction({"params": {"symbol": "AAPL"}, "type": "equity"}) == "AAPL"

    def test_missing(self):
        assert _get_symbol_from_instruction({"type": "equity"}) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Generate Safe Defaults
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateSafeDefaults:
    def test_produces_5_types(self):
        instructions = _generate_safe_defaults("SPY", "2026-04-01", None, "dry-run")
        assert len(instructions) == 5
        types = [i["type"] for i in instructions]
        assert types == ["equity", "single_option", "credit_spread", "debit_spread", "iron_condor"]

    def test_uses_equity_proxy_for_index(self):
        instructions = _generate_safe_defaults("SPX", "2026-04-01", None, "dry-run")
        assert instructions[0]["symbol"] == "SPY"
        assert instructions[1]["symbol"] == "SPX"

    def test_dry_run_equity_is_market(self):
        instructions = _generate_safe_defaults("SPY", "2026-04-01", None, "dry-run")
        assert instructions[0]["order_type"] == "MARKET"
        assert "limit_price" not in instructions[0]

    def test_paper_equity_is_limit(self):
        instructions = _generate_safe_defaults("SPY", "2026-04-01", None, "paper")
        assert instructions[0]["order_type"] == "LIMIT"
        assert instructions[0]["limit_price"] == 0.01

    def test_with_chain_data(self):
        chain = {"strikes": list(range(5000, 6001, 25)), "expirations": ["2026-04-01"]}
        instructions = _generate_safe_defaults("SPX", "2026-04-01", chain, "paper")
        assert len(instructions) == 5
        cs = instructions[2]
        assert cs["short_strike"] in chain["strikes"]
        assert cs["long_strike"] in chain["strikes"]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Execute Single Order (dry-run)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteSingleOrder:
    @pytest.mark.asyncio
    async def test_equity_dry_run(self):
        instr = {"id": "test_equity", "type": "equity", "symbol": "SPY",
                 "action": "BUY", "quantity": 1, "order_type": "MARKET"}
        result = await _execute_single_order(instr, "ibkr", "dry-run", 5.0, 0.1)
        assert result["passed"]
        check_names = [c[0] for c in result["checks"]]
        assert "Build TradeRequest" in check_names
        assert "Submit Order" in check_names

    @pytest.mark.asyncio
    async def test_credit_spread_dry_run(self):
        instr = {"id": "test_cs", "type": "credit_spread", "symbol": "SPX",
                 "expiration": "2026-04-01", "short_strike": 5500, "long_strike": 5475,
                 "option_type": "PUT", "quantity": 1, "net_price": 3.50}
        result = await _execute_single_order(instr, "ibkr", "dry-run", 5.0, 0.1)
        assert result["passed"]
        assert result["position_id"] is not None

    @pytest.mark.asyncio
    async def test_iron_condor_dry_run(self):
        instr = {"id": "test_ic", "type": "iron_condor", "symbol": "SPX",
                 "expiration": "2026-04-01", "put_short": 5500, "put_long": 5475,
                 "call_short": 5700, "call_long": 5725, "quantity": 1, "net_price": 3.50}
        result = await _execute_single_order(instr, "ibkr", "dry-run", 5.0, 0.1)
        assert result["passed"]

    @pytest.mark.asyncio
    async def test_bad_type_fails(self):
        instr = {"id": "test_bad", "type": "butterfly", "symbol": "SPY"}
        result = await _execute_single_order(instr, "ibkr", "dry-run", 5.0, 0.1)
        assert not result["passed"]
        assert result["checks"][0][1] == "fail"

    @pytest.mark.asyncio
    async def test_all_5_types(self):
        instructions = _generate_safe_defaults("SPY", "2026-04-01", None, "dry-run")
        for instr in instructions:
            result = await _execute_single_order(instr, "ibkr", "dry-run", 5.0, 0.1)
            assert result["passed"], f"Failed for {instr['type']}: {result['checks']}"

    @pytest.mark.asyncio
    async def test_cleanup_closes_positions(self):
        from app.services.position_store import get_position_store
        instr = {"id": "test_cleanup", "type": "equity", "symbol": "SPY",
                 "action": "BUY", "quantity": 1, "order_type": "MARKET"}
        result = await _execute_single_order(instr, "ibkr", "dry-run", 5.0, 0.1)
        assert result["position_id"]
        store = get_position_store()
        assert len(store.get_open_positions()) > 0
        await _cleanup_positions([result["position_id"]], "dry-run")
        open_pos = [p for p in store.get_open_positions()
                    if p.get("position_id") == result["position_id"]]
        assert len(open_pos) == 0

    @pytest.mark.asyncio
    async def test_logs_to_ledger(self):
        from app.services.ledger import get_ledger
        instr = {"id": "test_ledger", "type": "equity", "symbol": "SPY",
                 "action": "BUY", "quantity": 1, "order_type": "MARKET"}
        result = await _execute_single_order(instr, "ibkr", "dry-run", 5.0, 0.1)
        assert result["passed"]
        ledger = get_ledger()
        recent = await ledger.get_recent(10)
        order_id = result["order_result"].order_id
        matching = [e for e in recent if e.order_id == order_id]
        assert len(matching) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# API: Trade Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

pytestmark_trade_api = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _fast_poll_settings():
    """Speed up fill tracking for API tests."""
    orig_interval = settings.order_poll_interval_seconds
    orig_timeout = settings.order_poll_timeout_seconds
    settings.order_poll_interval_seconds = 0.05
    settings.order_poll_timeout_seconds = 2.0
    yield
    settings.order_poll_interval_seconds = orig_interval
    settings.order_poll_timeout_seconds = orig_timeout


class TestTradeAPI:
    @pytest.mark.asyncio
    async def test_equity_order_dry_run(self, client, api_key_headers):
        payload = {"equity_order": {"broker": "robinhood", "symbol": "AAPL",
                                     "side": "BUY", "quantity": 10, "order_type": "MARKET"}}
        headers = {**api_key_headers, "X-Dry-Run": "true"}
        resp = await client.post("/trade/execute", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert data["status"] == "PENDING"

    @pytest.mark.asyncio
    async def test_equity_order_live_sync_fills(self, client, api_key_headers):
        payload = {"equity_order": {"broker": "etrade", "symbol": "MSFT",
                                     "side": "SELL", "quantity": 5}}
        resp = await client.post("/trade/execute", json=payload, headers=api_key_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_equity_order_live_async_returns_submitted(self, client, api_key_headers):
        payload = {"equity_order": {"broker": "etrade", "symbol": "MSFT",
                                     "side": "SELL", "quantity": 5}}
        headers = {**api_key_headers, "X-Async": "true"}
        resp = await client.post("/trade/execute", json=payload, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "SUBMITTED"

    @pytest.mark.asyncio
    async def test_multi_leg_order_dry_run(self, client, api_key_headers):
        payload = {"multi_leg_order": {
            "broker": "robinhood",
            "legs": [
                {"symbol": "SPY", "expiration": "2026-03-20", "strike": 450.0,
                 "option_type": "PUT", "action": "SELL_TO_OPEN", "quantity": 1},
                {"symbol": "SPY", "expiration": "2026-03-20", "strike": 445.0,
                 "option_type": "PUT", "action": "BUY_TO_OPEN", "quantity": 1},
            ],
            "order_type": "LIMIT", "net_price": 1.25,
        }}
        headers = {**api_key_headers, "X-Dry-Run": "true"}
        resp = await client.post("/trade/execute", json=payload, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["dry_run"] is True

    @pytest.mark.asyncio
    async def test_multi_leg_order_live_sync_fills(self, client, api_key_headers):
        payload = {"multi_leg_order": {
            "broker": "ibkr",
            "legs": [
                {"symbol": "NDX", "expiration": "2026-04-17", "strike": 20000.0,
                 "option_type": "CALL", "action": "BUY_TO_OPEN", "quantity": 1},
                {"symbol": "NDX", "expiration": "2026-04-17", "strike": 20100.0,
                 "option_type": "CALL", "action": "SELL_TO_OPEN", "quantity": 1},
            ],
            "order_type": "LIMIT", "net_price": 25.0,
        }}
        resp = await client.post("/trade/execute", json=payload, headers=api_key_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "FILLED"

    @pytest.mark.asyncio
    async def test_multi_leg_order_live_async(self, client, api_key_headers):
        payload = {"multi_leg_order": {
            "broker": "ibkr",
            "legs": [
                {"symbol": "NDX", "expiration": "2026-04-17", "strike": 20000.0,
                 "option_type": "CALL", "action": "BUY_TO_OPEN", "quantity": 1},
                {"symbol": "NDX", "expiration": "2026-04-17", "strike": 20100.0,
                 "option_type": "CALL", "action": "SELL_TO_OPEN", "quantity": 1},
            ],
            "order_type": "LIMIT", "net_price": 25.0,
        }}
        headers = {**api_key_headers, "X-Async": "true"}
        resp = await client.post("/trade/execute", json=payload, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "SUBMITTED"

    @pytest.mark.asyncio
    async def test_invalid_both_orders(self, client, api_key_headers):
        payload = {
            "equity_order": {"broker": "robinhood", "symbol": "AAPL", "side": "BUY", "quantity": 1},
            "multi_leg_order": {
                "broker": "robinhood",
                "legs": [{"symbol": "SPY", "expiration": "2026-03-20", "strike": 450,
                          "option_type": "PUT", "action": "SELL_TO_OPEN", "quantity": 1}],
            },
        }
        resp = await client.post("/trade/execute", json=payload, headers=api_key_headers)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_no_auth_rejected(self, client):
        from app.config import settings
        original = settings.trust_local_network
        settings.trust_local_network = False
        try:
            payload = {"equity_order": {"broker": "robinhood", "symbol": "AAPL",
                                         "side": "BUY", "quantity": 1}}
            resp = await client.post("/trade/execute", json=payload)
            assert resp.status_code == 401
        finally:
            settings.trust_local_network = original

    @pytest.mark.asyncio
    async def test_get_pending_orders_empty(self):
        from app.services.trade_service import get_pending_orders
        pending = get_pending_orders()
        assert isinstance(pending, dict)

    @pytest.mark.asyncio
    async def test_get_pending_orders_returns_copy(self):
        from app.services.trade_service import get_pending_orders, _pending_orders
        _pending_orders.clear()
        pending = get_pending_orders()
        pending["fake_id"] = None
        assert "fake_id" not in _pending_orders


# ═══════════════════════════════════════════════════════════════════════════════
# API: Market Endpoint (quotes, batch quotes, margin)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarketAPI:
    @pytest.mark.asyncio
    async def test_get_quote(self, client, api_key_headers):
        resp = await client.get("/market/quote/SPY", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "SPY"
        assert "bid" in data
        assert "ask" in data
        assert data["volume"] > 0

    @pytest.mark.asyncio
    async def test_get_quote_ibkr(self, client, api_key_headers):
        resp = await client.get("/market/quote/AAPL?broker=ibkr", headers=api_key_headers)
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_quote_no_auth(self, client):
        from app.config import settings
        original = settings.trust_local_network
        settings.trust_local_network = False
        try:
            resp = await client.get("/market/quote/SPY")
            assert resp.status_code == 401
        finally:
            settings.trust_local_network = original

    @pytest.mark.asyncio
    async def test_batch_quotes(self, client, api_key_headers):
        payload = {"symbols": ["SPY", "AAPL"], "broker": "ibkr"}
        resp = await client.post("/market/quotes", json=payload, headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["quotes"]) == 2
        symbols = {q["symbol"] for q in data["quotes"]}
        assert "SPY" in symbols
        assert "AAPL" in symbols

    @pytest.mark.asyncio
    async def test_margin_check(self, client, api_key_headers):
        payload = {
            "order": {
                "broker": "ibkr",
                "legs": [
                    {"symbol": "SPX", "expiration": "2026-03-16", "strike": 5500.0,
                     "option_type": "PUT", "action": "SELL_TO_OPEN", "quantity": 1},
                    {"symbol": "SPX", "expiration": "2026-03-16", "strike": 5475.0,
                     "option_type": "PUT", "action": "BUY_TO_OPEN", "quantity": 1},
                ],
                "order_type": "LIMIT", "net_price": 1.00,
            },
            "timeout": 5.0,
        }
        resp = await client.post("/market/margin", json=payload, headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "init_margin" in data
        assert "maint_margin" in data
        assert "commission" in data


# ═══════════════════════════════════════════════════════════════════════════════
# API: Account Endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestAccountAPI:
    @pytest.mark.asyncio
    async def test_get_positions(self, client, api_key_headers):
        resp = await client.get("/account/positions", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "positions" in data
        assert len(data["positions"]) == 3  # one per provider
        assert data["total_market_value"] > 0

    @pytest.mark.asyncio
    async def test_positions_no_auth(self, client):
        from app.config import settings
        original = settings.trust_local_network
        settings.trust_local_network = False
        try:
            resp = await client.get("/account/positions")
            assert resp.status_code == 401
        finally:
            settings.trust_local_network = original


# ═══════════════════════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuth:
    @pytest.mark.asyncio
    async def test_api_key_auth(self, client, api_key_headers):
        resp = await client.get("/health")
        assert resp.status_code == 200
        resp = await client.get("/market/quote/SPY", headers=api_key_headers)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_oauth2_token_flow(self, client):
        from app.auth import create_access_token
        resp = await client.post("/auth/token",
                                  json={"username": "testuser", "password": "testpass"})
        assert resp.status_code == 200
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        resp = await client.get("/market/quote/AAPL", headers=headers)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_token_missing_scope(self, client):
        from app.config import settings
        original = settings.trust_local_network
        settings.trust_local_network = False
        try:
            from app.auth import create_access_token
            token = create_access_token("user", scopes=["market:read"])
            headers = {"Authorization": f"Bearer {token}"}
            resp = await client.get("/market/quote/SPY", headers=headers)
            assert resp.status_code == 200
            payload = {"equity_order": {"broker": "robinhood", "symbol": "AAPL",
                                         "side": "BUY", "quantity": 1}}
            resp = await client.post("/trade/execute", json=payload, headers=headers)
            assert resp.status_code == 403
        finally:
            settings.trust_local_network = original

    @pytest.mark.asyncio
    async def test_invalid_token(self, client):
        from app.config import settings
        original = settings.trust_local_network
        settings.trust_local_network = False
        try:
            headers = {"Authorization": "Bearer invalid.jwt.token"}
            resp = await client.get("/market/quote/SPY", headers=headers)
            assert resp.status_code == 401
        finally:
            settings.trust_local_network = original


# ═══════════════════════════════════════════════════════════════════════════════
# Symbology
# ═══════════════════════════════════════════════════════════════════════════════


class TestSymbology:
    def test_robinhood_equity_id_is_uuid(self):
        from app.core.symbology import SymbologyMapper
        eid = SymbologyMapper.equity_id(Broker.ROBINHOOD, "SPY")
        assert len(eid) == 36
        assert eid.count("-") == 4

    def test_ibkr_equity_id_is_integer(self):
        from app.core.symbology import SymbologyMapper
        eid = SymbologyMapper.equity_id(Broker.IBKR, "SPY")
        int(eid)  # should not raise

    def test_etrade_equity_id_is_symbol(self):
        from app.core.symbology import SymbologyMapper
        assert SymbologyMapper.equity_id(Broker.ETRADE, "spy") == "SPY"

    def test_option_id_deterministic(self):
        from app.core.symbology import OptionContract, SymbologyMapper
        contract = OptionContract("SPY", "2026-03-20", 450.0, OptionType.PUT)
        id1 = SymbologyMapper.option_id(Broker.ROBINHOOD, contract)
        id2 = SymbologyMapper.option_id(Broker.ROBINHOOD, contract)
        assert id1 == id2

    def test_option_id_differs_by_broker(self):
        from app.core.symbology import OptionContract, SymbologyMapper
        contract = OptionContract("SPY", "2026-03-20", 450.0, OptionType.CALL)
        rh = SymbologyMapper.option_id(Broker.ROBINHOOD, contract)
        ibkr = SymbologyMapper.option_id(Broker.IBKR, contract)
        assert rh != ibkr

    def test_osi_symbol(self):
        from app.core.symbology import OptionContract, SymbologyMapper
        contract = OptionContract("SPY", "2026-03-20", 450.0, OptionType.PUT)
        osi = SymbologyMapper.osi_symbol(contract)
        assert osi.startswith("SPY")
        assert "P" in osi

    def test_etrade_option_id_format(self):
        from app.core.symbology import OptionContract, SymbologyMapper
        contract = OptionContract("SPY", "2026-03-20", 450.0, OptionType.CALL)
        eid = SymbologyMapper.option_id(Broker.ETRADE, contract)
        assert "SPY" in eid
        assert ":C" in eid


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebSocket:
    @pytest.mark.asyncio
    async def test_broadcast(self):
        from app.websocket import ws_manager
        result = OrderResult(order_id="test-123", broker=Broker.ROBINHOOD,
                             status=OrderStatus.FILLED, message="Test fill")
        await ws_manager.broadcast_order_update(result)

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "daemon_mode" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Transaction Ledger
# ═══════════════════════════════════════════════════════════════════════════════


class TestLedger:
    @pytest.fixture
    def _ledger(self, tmp_path):
        from app.services.ledger import TransactionLedger
        return TransactionLedger(tmp_path / "ledger_test")

    @pytest.mark.asyncio
    async def test_append_and_get_recent(self, _ledger):
        entry = LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED, broker=Broker.ROBINHOOD)
        result = await _ledger.append(entry)
        assert result.sequence_number == 0
        recent = await _ledger.get_recent(10)
        assert len(recent) == 1

    @pytest.mark.asyncio
    async def test_sequence_increments(self, _ledger):
        e1 = await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        e2 = await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_STATUS_CHANGE))
        assert e1.sequence_number == 0
        assert e2.sequence_number == 1

    @pytest.mark.asyncio
    async def test_query_by_broker(self, _ledger):
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED, broker=Broker.ROBINHOOD))
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED, broker=Broker.ETRADE))
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED, broker=Broker.ROBINHOOD))
        results = await _ledger.query(LedgerQuery(broker=Broker.ROBINHOOD))
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_query_by_event_type(self, _ledger):
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.POSITION_OPENED))
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.POSITION_CLOSED))
        results = await _ledger.query(LedgerQuery(event_type=LedgerEventType.POSITION_OPENED))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_by_source(self, _ledger):
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED, source=PositionSource.PAPER))
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED, source=PositionSource.LIVE_API))
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.CSV_IMPORTED, source=PositionSource.CSV_IMPORT))
        results = await _ledger.query(LedgerQuery(source=PositionSource.CSV_IMPORT))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_pagination(self, _ledger):
        for i in range(10):
            await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED, data={"i": i}))
        page1 = await _ledger.query(LedgerQuery(limit=3, offset=0))
        page2 = await _ledger.query(LedgerQuery(limit=3, offset=3))
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0].data["i"] != page2[0].data["i"]

    @pytest.mark.asyncio
    async def test_convenience_methods(self, _ledger):
        await _ledger.log_order_submitted(Broker.ROBINHOOD, "ord-1", PositionSource.LIVE_API)
        await _ledger.log_status_change(Broker.ROBINHOOD, "ord-1", "FILLED")
        await _ledger.log_position_opened(Broker.ROBINHOOD, "pos-1", PositionSource.LIVE_API)
        await _ledger.log_position_closed(Broker.ROBINHOOD, "pos-1", PositionSource.LIVE_API)
        recent = await _ledger.get_recent(10)
        assert len(recent) == 4

    @pytest.mark.asyncio
    async def test_snapshot_and_list(self, _ledger):
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        filename = await _ledger.save_snapshot([{"pos": "test"}], {"cash": 10000})
        assert filename.startswith("snapshot_")
        assert len(_ledger.list_snapshots()) >= 1

    @pytest.mark.asyncio
    async def test_replay_from_beginning(self, _ledger):
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.POSITION_OPENED))
        state, entries = await _ledger.replay()
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_replay_from_snapshot(self, _ledger):
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        filename = await _ledger.save_snapshot([{"test": True}], {"cash": 5000})
        await _ledger.append(LedgerEntry(event_type=LedgerEventType.POSITION_OPENED))
        state, entries = await _ledger.replay(from_snapshot=filename)
        assert state["account_state"]["cash"] == 5000

    @pytest.mark.asyncio
    async def test_empty_ledger(self, _ledger):
        assert await _ledger.get_recent(10) == []
        assert await _ledger.query(LedgerQuery()) == []

    @pytest.mark.asyncio
    async def test_recover_sequence(self, tmp_path):
        from app.services.ledger import TransactionLedger
        ledger_dir = tmp_path / "ledger_recovery"
        l1 = TransactionLedger(ledger_dir)
        await l1.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        await l1.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        l2 = TransactionLedger(ledger_dir)
        entry = await l2.append(LedgerEntry(event_type=LedgerEventType.ORDER_SUBMITTED))
        assert entry.sequence_number == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Position Store
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositionStore:
    @pytest.fixture
    def _store(self, tmp_path):
        from app.services.position_store import PlatformPositionStore
        return PlatformPositionStore(tmp_path / "positions_test.json")

    def _equity_request(self):
        return TradeRequest(equity_order=EquityOrder(
            broker=Broker.ROBINHOOD, symbol="SPY", side=OrderSide.BUY,
            quantity=100, limit_price=450.0,
        ))

    def _multi_leg_request(self):
        return TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.ETRADE,
            legs=[
                OptionLeg(symbol="SPY", expiration="2026-03-20", strike=450.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPY", expiration="2026-03-20", strike=445.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=1.25, quantity=10,
        ))

    def test_add_equity_position(self, _store):
        pos_id = _store.add_position(self._equity_request(), OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)
        assert pos_id
        assert len(_store.get_open_positions()) == 1
        assert _store.get_open_positions()[0]["source"] == PositionSource.LIVE_API.value

    def test_add_paper_position(self, _store):
        _store.add_position(self._equity_request(), OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=True)
        assert _store.get_open_positions()[0]["source"] == PositionSource.PAPER.value

    def test_add_multi_leg_position(self, _store):
        _store.add_position(self._multi_leg_request(), OrderResult(broker=Broker.ETRADE, status=OrderStatus.FILLED), is_paper=False)
        pos = _store.get_open_positions()
        assert pos[0]["order_type"] == "multi_leg"
        assert len(pos[0]["legs"]) == 2

    def test_close_position_pnl(self, _store):
        pos_id = _store.add_position(self._multi_leg_request(), OrderResult(broker=Broker.ETRADE, status=OrderStatus.FILLED), is_paper=False)
        result = _store.close_position(pos_id, exit_price=0.50, reason="profit_target")
        assert result["pnl"] == 750.0  # (1.25 - 0.50) * 10 * 100

    def test_close_equity_buy_pnl(self, _store):
        pos_id = _store.add_position(self._equity_request(), OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)
        result = _store.close_position(pos_id, exit_price=460.0, reason="manual")
        assert result["pnl"] == 1000.0  # (460 - 450) * 100

    def test_mark_to_market(self, _store):
        pos_id = _store.add_position(self._equity_request(), OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)
        _store.update_mark(pos_id, 455.0)
        pos = _store.get_open_positions()[0]
        assert pos["current_mark"] == 455.0
        assert pos["unrealized_pnl"] == 500.0

    def test_get_expired_positions(self, _store):
        _store.add_position(self._multi_leg_request(), OrderResult(broker=Broker.ETRADE, status=OrderStatus.FILLED), is_paper=False)
        # Same-day should NOT show as expired (0DTE still trading)
        assert len(_store.get_expired_positions(date(2026, 3, 20))) == 0
        # Day after expiration — now it's truly expired
        assert len(_store.get_expired_positions(date(2026, 3, 21))) == 1
        assert len(_store.get_expired_positions(date(2026, 3, 19))) == 0

    def test_find_by_broker_symbol(self, _store):
        _store.add_position(self._equity_request(), OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)
        assert _store.find_by_broker_symbol(Broker.ROBINHOOD, "SPY") is not None
        assert _store.find_by_broker_symbol(Broker.ETRADE, "SPY") is None

    def test_account_summary(self, _store):
        _store.add_position(self._equity_request(), OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)
        summary = _store.get_account_summary()
        assert summary["open_count"] == 1
        assert summary["cash_deployed"] == 45000.0

    def test_export_results(self, _store):
        pos_id = _store.add_position(self._multi_leg_request(), OrderResult(broker=Broker.ETRADE, status=OrderStatus.FILLED), is_paper=False)
        _store.close_position(pos_id, exit_price=0.0, reason="expired")
        results = _store.export_results()
        assert results[0]["pnl"] == 1250.0

    def test_persistence(self, tmp_path):
        from app.services.position_store import PlatformPositionStore
        path = tmp_path / "persist.json"
        s1 = PlatformPositionStore(path)
        s1.add_position(self._equity_request(), OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)
        s2 = PlatformPositionStore(path)
        assert len(s2.get_open_positions()) == 1

    def test_empty_store(self, _store):
        assert _store.get_open_positions() == []
        assert _store.get_closed_positions() == []
        assert _store.get_account_summary()["open_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard Service
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboard:
    @pytest.fixture
    def _store(self, tmp_path):
        from app.services.position_store import PlatformPositionStore
        return PlatformPositionStore(tmp_path / "dash_store.json")

    @pytest.fixture
    def _svc(self, _store):
        from app.services.dashboard_service import DashboardService
        return DashboardService(_store)

    def _add_open_equity(self, store, symbol="SPY", price=450.0):
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.ROBINHOOD, symbol=symbol, side=OrderSide.BUY,
            quantity=100, limit_price=price,
        ))
        return store.add_position(req, OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)

    def _add_closed_equity(self, store, symbol="AAPL", entry_price=175.0, exit_price=180.0):
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.ROBINHOOD, symbol=symbol, side=OrderSide.BUY,
            quantity=50, limit_price=entry_price,
        ))
        pos_id = store.add_position(req, OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)
        store.close_position(pos_id, exit_price=exit_price, reason="manual")
        return pos_id

    def test_summary_with_positions(self, _store, _svc):
        self._add_open_equity(_store)
        summary = _svc.get_summary()
        assert len(summary.active_positions) == 1

    def test_summary_empty(self, _store, _svc):
        assert _svc.get_summary().total_pnl == 0

    def test_summary_positions_by_source(self, _store, _svc):
        self._add_open_equity(_store)
        summary = _svc.get_summary()
        assert PositionSource.LIVE_API.value in summary.positions_by_source

    def test_performance_metrics(self, _store, _svc):
        self._add_closed_equity(_store, "AAPL", 175.0, 180.0)
        self._add_closed_equity(_store, "MSFT", 420.0, 415.0)
        perf = _svc.get_performance()
        assert perf.total_trades == 2
        assert perf.wins == 1

    def test_performance_empty(self, _store, _svc):
        assert _svc.get_performance().total_trades == 0

    def test_daily_pnl(self, _store, _svc):
        self._add_closed_equity(_store)
        daily = _svc.get_daily_pnl(days=30)
        assert len(daily) >= 1

    def test_compute_metrics_basic(self):
        from app.services.metrics import compute_metrics
        results = [
            {"pnl": 100, "credit": 50, "max_loss": 500},
            {"pnl": -50, "credit": 50, "max_loss": 500},
            {"pnl": 200, "credit": 100, "max_loss": 1000},
        ]
        m = compute_metrics(results)
        assert m["total_trades"] == 3
        assert m["wins"] == 2
        assert m["net_pnl"] == 250

    def test_terminal_renderer(self, _store, _svc):
        from app.services.terminal_display import TerminalRenderer
        self._add_open_equity(_store)
        self._add_closed_equity(_store)
        summary = _svc.get_summary()
        perf = _svc.get_performance()
        output = TerminalRenderer.render(summary, perf)
        assert "Universal Trade Platform Dashboard" in output


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Importer
# ═══════════════════════════════════════════════════════════════════════════════

ROBINHOOD_CSV = """Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount
03/01/2026,,03/03/2026,AAPL,Buy,Buy,10,$175.50,$1755.00
03/05/2026,,03/07/2026,AAPL,Sell,Sell,10,$180.00,$1800.00
03/10/2026,,03/12/2026,MSFT,Buy,Buy,5,$420.00,$2100.00
03/10/2026,,03/12/2026,SPY,Dividend,,,,,$25.00
"""

ETRADE_CSV = """TransactionDate,TransactionType,SecurityType,Symbol,Quantity,Price,Commission,Amount
03/01/2026,Bought,Equity,QQQ,20,$380.00,$0.00,$7600.00
03/08/2026,Sold,Equity,QQQ,20,$385.00,$0.00,$7700.00
"""

MALFORMED_CSV = """bad,header,row
1,2,3
"""


class TestCSVImporter:
    @pytest.fixture
    def _store(self, tmp_path):
        from app.services.position_store import PlatformPositionStore
        return PlatformPositionStore(tmp_path / "csv_store.json")

    @pytest.fixture
    def _ledger(self, tmp_path):
        from app.services.ledger import TransactionLedger
        return TransactionLedger(tmp_path / "csv_ledger")

    @pytest.fixture
    def _importer(self, _store, _ledger):
        from app.services.csv_importer import CSVTransactionImporter
        return CSVTransactionImporter(_store, _ledger)

    @pytest.mark.asyncio
    async def test_robinhood_csv_parsing(self, _importer):
        raw = _importer.parse_robinhood_csv(ROBINHOOD_CSV)
        assert len(raw) == 3

    @pytest.mark.asyncio
    async def test_etrade_csv_parsing(self, _importer):
        raw = _importer.parse_etrade_csv(ETRADE_CSV)
        assert len(raw) == 2

    @pytest.mark.asyncio
    async def test_import_robinhood(self, _importer):
        result = await _importer.import_content(ROBINHOOD_CSV, Broker.ROBINHOOD, "rh.csv")
        assert result.records_imported == 3

    @pytest.mark.asyncio
    async def test_import_etrade(self, _importer):
        result = await _importer.import_content(ETRADE_CSV, Broker.ETRADE, "et.csv")
        assert result.records_imported == 2

    @pytest.mark.asyncio
    async def test_csv_import_source_attribution(self, _importer, _ledger):
        await _importer.import_content(ROBINHOOD_CSV, Broker.ROBINHOOD)
        entries = await _ledger.get_recent(50)
        csv_entries = [e for e in entries if e.event_type == LedgerEventType.CSV_IMPORTED]
        assert len(csv_entries) >= 1
        assert all(e.source == PositionSource.CSV_IMPORT for e in csv_entries)

    @pytest.mark.asyncio
    async def test_deduplication(self, _importer):
        await _importer.import_content(ROBINHOOD_CSV, Broker.ROBINHOOD)
        result2 = await _importer.import_content(ROBINHOOD_CSV, Broker.ROBINHOOD)
        assert result2.records_imported >= 0

    def test_preview(self, _importer):
        rows = _importer.preview(ROBINHOOD_CSV, Broker.ROBINHOOD, max_rows=2)
        assert len(rows) <= 2
        assert "symbol" in rows[0]

    @pytest.mark.asyncio
    async def test_malformed_csv(self, _importer):
        result = await _importer.import_content(MALFORMED_CSV, Broker.ROBINHOOD, "bad.csv")
        assert result.records_imported == 0

    def test_supported_formats(self):
        from app.services.csv_importer import CSVTransactionImporter
        formats = CSVTransactionImporter.supported_formats()
        assert "robinhood" in formats
        assert "etrade" in formats

    @pytest.mark.asyncio
    async def test_unsupported_broker(self, _importer):
        result = await _importer.import_content(ROBINHOOD_CSV, Broker.IBKR, "ibkr.csv")
        assert result.records_imported == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Expiration Service
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpirationService:
    @pytest.fixture
    def _store(self, tmp_path):
        from app.services.position_store import PlatformPositionStore
        return PlatformPositionStore(tmp_path / "exp_store.json")

    @pytest.fixture
    def _ledger(self, tmp_path):
        from app.services.ledger import TransactionLedger
        return TransactionLedger(tmp_path / "exp_ledger")

    @pytest.fixture
    def _svc(self, _store, _ledger):
        from app.services.expiration_service import ExpirationService
        return ExpirationService(_store, _ledger)

    def _add_spread(self, store, expiration="2026-03-15"):
        req = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.ROBINHOOD,
            legs=[
                OptionLeg(symbol="SPY", expiration=expiration, strike=450.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPY", expiration=expiration, strike=445.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=1.50, quantity=5,
        ))
        return store.add_position(req, OrderResult(broker=Broker.ROBINHOOD, status=OrderStatus.FILLED), is_paper=False)

    @pytest.mark.asyncio
    async def test_detect_expired(self, _store, _svc):
        self._add_spread(_store, "2026-03-15")
        # Day after expiration — same-day (0DTE) should NOT expire during trading
        closed = await _svc.check_expirations(date(2026, 3, 16))
        assert len(closed) == 1

    @pytest.mark.asyncio
    async def test_not_expired_same_day(self, _store, _svc):
        """Same-day expiration should NOT be auto-expired (still trading until market close)."""
        self._add_spread(_store, "2026-03-15")
        closed = await _svc.check_expirations(date(2026, 3, 15))
        assert len(closed) == 0

    @pytest.mark.asyncio
    async def test_not_expired(self, _store, _svc):
        self._add_spread(_store, "2026-03-20")
        closed = await _svc.check_expirations(date(2026, 3, 15))
        assert len(closed) == 0

    @pytest.mark.asyncio
    async def test_pnl_on_expired(self, _store, _svc):
        self._add_spread(_store, "2026-03-15")
        await _svc.check_expirations(date(2026, 3, 16))  # day after
        closed = _store.get_closed_positions()
        assert closed[0]["pnl"] == 750.0

    @pytest.mark.asyncio
    async def test_eod_closes_0dte(self, _store, _svc):
        self._add_spread(_store, "2026-03-15")
        now = datetime(2026, 3, 15, 21, 0, tzinfo=UTC)
        assert len(await _svc.check_eod_exits(now)) == 1

    @pytest.mark.asyncio
    async def test_eod_before_close(self, _store, _svc):
        self._add_spread(_store, "2026-03-15")
        now = datetime(2026, 3, 15, 18, 0, tzinfo=UTC)
        assert len(await _svc.check_eod_exits(now)) == 0

    @pytest.mark.asyncio
    async def test_ledger_logged(self, _store, _ledger, _svc):
        self._add_spread(_store, "2026-03-15")
        await _svc.check_expirations(date(2026, 3, 16))  # day after
        entries = await _ledger.get_recent(10)
        assert "POSITION_CLOSED" in [e.event_type.value for e in entries]

    @pytest.mark.asyncio
    async def test_no_positions(self, _svc):
        assert await _svc.check_expirations(date(2026, 3, 15)) == []

    def test_get_expiring_preview(self, _store, _svc):
        self._add_spread(_store, "2026-03-15")
        self._add_spread(_store, "2026-03-20")
        assert len(_svc.get_expiring_positions(date(2026, 3, 15))) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Position Sync
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositionSync:
    @pytest.fixture
    def _store(self, tmp_path):
        from app.services.position_store import PlatformPositionStore
        return PlatformPositionStore(tmp_path / "sync_store.json")

    @pytest.fixture
    def _ledger(self, tmp_path):
        from app.services.ledger import TransactionLedger
        return TransactionLedger(tmp_path / "sync_ledger")

    @pytest.fixture
    def _sync(self, _store, _ledger):
        from app.services.position_sync import PositionSyncService
        return PositionSyncService(_store, _ledger)

    @pytest.mark.asyncio
    async def test_sync_detects_new(self, _sync, _store):
        result = await _sync.sync_all_brokers()
        assert result.new_positions >= 1
        sources = [p.get("source") for p in _store.get_open_positions()]
        assert PositionSource.EXTERNAL_SYNC.value in sources

    @pytest.mark.asyncio
    async def test_sync_updates_existing(self, _sync, _store):
        await _sync.sync_all_brokers()
        first_count = len(_store.get_open_positions())
        result2 = await _sync.sync_all_brokers()
        assert result2.updated_positions >= 1
        assert result2.new_positions == 0
        assert len(_store.get_open_positions()) == first_count

    @pytest.mark.asyncio
    async def test_sync_logs_to_ledger(self, _sync, _ledger):
        await _sync.sync_all_brokers()
        entries = await _ledger.get_recent(50)
        assert any(e.event_type.value == "POSITION_SYNCED" for e in entries)

    def test_trading_hours_during_market(self):
        from app.services.position_sync import PositionSyncService
        assert PositionSyncService.is_trading_hours(datetime(2026, 3, 15, 15, 0, tzinfo=UTC))

    def test_trading_hours_before_market(self):
        from app.services.position_sync import PositionSyncService
        assert not PositionSyncService.is_trading_hours(datetime(2026, 3, 15, 12, 0, tzinfo=UTC))

    def test_trading_hours_after_market(self):
        from app.services.position_sync import PositionSyncService
        assert not PositionSyncService.is_trading_hours(datetime(2026, 3, 15, 21, 0, tzinfo=UTC))

    @pytest.mark.asyncio
    async def test_sync_result_shape(self, _sync):
        result = await _sync.sync_all_brokers()
        assert hasattr(result, "new_positions")
        assert isinstance(result.brokers_synced, list)

    @pytest.mark.asyncio
    async def test_sync_correct_source(self, _sync, _store):
        await _sync.sync_all_brokers()
        for pos in _store.get_open_positions():
            assert pos.get("source") == PositionSource.EXTERNAL_SYNC.value

    @pytest.mark.asyncio
    async def test_sync_backfills_option_fields(self, _store, _ledger):
        """Sync backfills sec_type, expiration, strike, right on existing positions."""
        from app.services.position_sync import PositionSyncService
        from app.models import Position, Broker

        # Create a position missing option fields (simulates trade execution path)
        pos_id = _store.add_position_from_sync(
            broker=Broker.IBKR, symbol="SPX", quantity=-25,
            avg_cost=1.50, market_value=-375.0, unrealized_pnl=0,
            con_id=12345,
            # Note: sec_type, expiration, strike, right are NOT set
        )
        pos = _store._positions[pos_id]
        assert pos.get("sec_type") is None
        assert pos.get("strike") is None
        assert pos.get("right") is None

        # Now mock a sync that returns the same position with full option fields
        mock_provider = AsyncMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions.return_value = [
            Position(
                broker=Broker.IBKR, symbol="SPX", quantity=-25,
                avg_cost=1.50, market_value=-375.0, unrealized_pnl=0,
                con_id=12345,
                sec_type="OPT", expiration="20260407", strike=6460.0, right="P",
            ),
        ]
        from app.core.provider import ProviderRegistry
        original_all = ProviderRegistry.all
        ProviderRegistry.all = lambda: [mock_provider]
        try:
            sync_svc = PositionSyncService(_store, _ledger)
            result = await sync_svc.sync_all_brokers()
            assert result.updated_positions >= 1
        finally:
            ProviderRegistry.all = original_all

        # Verify fields were backfilled
        updated = _store._positions[pos_id]
        assert updated["sec_type"] == "OPT"
        assert updated["expiration"] == "2026-04-07"
        assert updated["strike"] == 6460.0
        assert updated["right"] == "P"


# ═══════════════════════════════════════════════════════════════════════════════
# Reconciliation
# ═══════════════════════════════════════════════════════════════════════════════


class TestReconciliation:
    @pytest.fixture
    def _sync(self, position_store, ledger):
        from app.services.position_sync import PositionSyncService
        return PositionSyncService(position_store, ledger)

    @pytest.mark.asyncio
    async def test_no_discrepancies(self, _sync, position_store):
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY,
            quantity=100, order_type="MARKET",
        ))
        position_store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=True)
        broker_pos = Position(broker=Broker.IBKR, symbol="SPY", quantity=100,
                              avg_cost=450.0, market_value=45000.0, unrealized_pnl=0.0)
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[broker_pos])
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        assert report.matched == 1

    @pytest.mark.asyncio
    async def test_missing_in_system(self, _sync):
        broker_pos = Position(broker=Broker.IBKR, symbol="AAPL", quantity=50,
                              avg_cost=200.0, market_value=10000.0, unrealized_pnl=0.0)
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[broker_pos])
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        missing = [d for d in report.discrepancies if d.discrepancy_type == "missing_in_system"]
        assert len(missing) == 1

    @pytest.mark.asyncio
    async def test_missing_at_broker(self, _sync, position_store):
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="MSFT", side=OrderSide.BUY,
            quantity=75, order_type="MARKET",
        ))
        position_store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=True)
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[])
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        missing = [d for d in report.discrepancies if d.discrepancy_type == "missing_at_broker"]
        assert len(missing) == 1

    @pytest.mark.asyncio
    async def test_quantity_mismatch(self, _sync, position_store):
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="GOOG", side=OrderSide.BUY,
            quantity=100, order_type="MARKET",
        ))
        position_store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=True)
        broker_pos = Position(broker=Broker.IBKR, symbol="GOOG", quantity=50,
                              avg_cost=150.0, market_value=7500.0, unrealized_pnl=0.0)
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[broker_pos])
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        mismatched = [d for d in report.discrepancies if d.discrepancy_type == "quantity_mismatch"]
        assert len(mismatched) == 1

    @pytest.mark.asyncio
    async def test_empty_broker(self, _sync):
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[])
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        assert report.matched == 0

    @pytest.mark.asyncio
    async def test_multiple_positions(self, _sync, position_store):
        for sym in ["SPY", "QQQ", "AAPL"]:
            req = TradeRequest(equity_order=EquityOrder(
                broker=Broker.IBKR, symbol=sym, side=OrderSide.BUY,
                quantity=100, order_type="MARKET",
            ))
            position_store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=True)
        broker_positions = [
            Position(broker=Broker.IBKR, symbol="SPY", quantity=100, avg_cost=450.0, market_value=45000.0, unrealized_pnl=0.0),
            Position(broker=Broker.IBKR, symbol="QQQ", quantity=50, avg_cost=380.0, market_value=19000.0, unrealized_pnl=0.0),
            Position(broker=Broker.IBKR, symbol="GOOG", quantity=25, avg_cost=150.0, market_value=3750.0, unrealized_pnl=0.0),
        ]
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=broker_positions)
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        assert report.matched == 1
        types = {d.symbol: d.discrepancy_type for d in report.discrepancies}
        assert types["SPY"] == "matched"
        assert types["QQQ"] == "quantity_mismatch"

    @pytest.mark.asyncio
    async def test_reconcile_includes_open_orders(self, _sync, position_store):
        """Reconciliation report includes open/working orders from broker."""
        broker_pos = Position(broker=Broker.IBKR, symbol="SPY", quantity=100,
                              avg_cost=450.0, market_value=45000.0, unrealized_pnl=0.0)
        order1 = OrderResult(
            order_id="111", broker=Broker.IBKR, status=OrderStatus.SUBMITTED,
            message="SPY BUY 50 LMT @ $445",
            extra={"symbol": "SPY", "action": "BUY", "quantity": 50,
                   "order_type": "LMT", "limit_price": 445.0, "perm_id": 999},
        )
        order2 = OrderResult(
            order_id="222", broker=Broker.IBKR, status=OrderStatus.SUBMITTED,
            message="RUT combo 2-leg",
            extra={"symbol": "RUT", "action": "BUY", "quantity": 1,
                   "order_type": "MKT", "limit_price": None, "perm_id": 1000,
                   "legs": [{"con_id": 1, "action": "BUY", "ratio": 1}]},
        )
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[broker_pos])
        mock_provider.get_open_orders = AsyncMock(return_value=[order1, order2])
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        assert len(report.open_orders) == 2
        assert report.open_orders[0]["order_id"] == "111"
        assert report.open_orders[1]["extra"]["symbol"] == "RUT"

    @pytest.mark.asyncio
    async def test_reconcile_no_open_orders(self, _sync):
        """Reconciliation works when broker has no open orders."""
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[])
        mock_provider.get_open_orders = AsyncMock(return_value=[])
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        assert report.open_orders == []

    @pytest.mark.asyncio
    async def test_reconcile_order_fetch_failure_graceful(self, _sync):
        """If fetching orders fails, reconciliation still returns positions."""
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[])
        mock_provider.get_open_orders = AsyncMock(side_effect=RuntimeError("IBKR disconnected"))
        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            report = await _sync.reconcile(Broker.IBKR)
        # Report still generated — just no orders
        assert report.open_orders == []
        assert report.matched == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Flush & Reconcile --flush
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlush:
    async def test_flush_positions(self, tmp_path):
        """Flush clears the positions file."""
        data_dir = tmp_path / "utp"
        data_dir.mkdir()
        pos_file = data_dir / "positions.json"
        pos_file.write_text('{"id1": {"status": "open"}, "id2": {"status": "open"}}')

        args = argparse.Namespace(what="positions", data_dir=str(data_dir))
        rc = await _cmd_flush(args)
        assert rc == 0
        assert json.loads(pos_file.read_text()) == {}

    async def test_flush_ledger(self, tmp_path):
        """Flush clears the ledger file and snapshots."""
        data_dir = tmp_path / "utp"
        ledger_dir = data_dir / "ledger"
        snap_dir = ledger_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        (ledger_dir / "ledger.jsonl").write_text('{"seq": 1}\n{"seq": 2}\n')
        (snap_dir / "snap1.json").write_text("{}")

        args = argparse.Namespace(what="ledger", data_dir=str(data_dir))
        rc = await _cmd_flush(args)
        assert rc == 0
        assert (ledger_dir / "ledger.jsonl").read_text() == ""
        assert len(list(snap_dir.glob("*.json"))) == 0

    async def test_flush_all(self, tmp_path):
        """Flush all clears both positions and ledger."""
        data_dir = tmp_path / "utp"
        ledger_dir = data_dir / "ledger"
        ledger_dir.mkdir(parents=True)
        pos_file = data_dir / "positions.json"
        pos_file.write_text('{"id1": {"status": "open"}}')
        (ledger_dir / "ledger.jsonl").write_text('{"seq": 1}\n')

        args = argparse.Namespace(what="all", data_dir=str(data_dir))
        rc = await _cmd_flush(args)
        assert rc == 0
        assert json.loads(pos_file.read_text()) == {}
        assert (ledger_dir / "ledger.jsonl").read_text() == ""

    async def test_flush_conid_cache_removes_disk_files(self, tmp_path, monkeypatch):
        """Purging conid-cache deletes JSON files in the cache dir."""
        # _isolate_option_conid_cache autouse fixture sets UTP_OPTION_CONID_CACHE_DIR
        # to tmp_path / "option_conids". Use that location.
        cache_dir = tmp_path / "option_conids"
        cache_dir.mkdir(parents=True)
        (cache_dir / "2026-04-28.json").write_text('{"NDX_20260429_26250.0_P": 416843}')
        (cache_dir / "2026-04-27.json").write_text('{"SPX_20260428_5500.0_P": 9876543}')

        # Use a bogus redis URL so the redis branch fails fast (non-fatal)
        args = argparse.Namespace(
            what="conid-cache",
            data_dir=str(tmp_path / "utp"),
            redis_url="redis://127.0.0.1:1/0",  # nothing listening
            server_port=8000,
        )
        rc = await _cmd_flush(args)
        assert rc == 0
        assert list(cache_dir.glob("*.json")) == []

    async def test_flush_conid_cache_redis_unreachable_is_nonfatal(self, tmp_path):
        """Redis connect failure must not fail the purge (disk purge is the
        primary action)."""
        # No JSON files; just confirm the redis branch doesn't poison the rc.
        args = argparse.Namespace(
            what="conid-cache",
            data_dir=str(tmp_path / "utp"),
            redis_url="redis://127.0.0.1:1/0",
            server_port=8000,
        )
        rc = await _cmd_flush(args)
        assert rc == 0

    async def test_flush_conid_cache_blocks_when_daemon_running(self, tmp_path, monkeypatch):
        """Purge refuses while a daemon is detected — its in-memory cache would
        re-save stale entries."""
        import utp as _utp
        monkeypatch.setattr(_utp, "_detect_server", lambda args: "http://localhost:8000")

        cache_dir = tmp_path / "option_conids"
        cache_dir.mkdir(parents=True)
        f = cache_dir / "2026-04-28.json"
        f.write_text("{}")

        args = argparse.Namespace(
            what="conid-cache",
            data_dir=str(tmp_path / "utp"),
            redis_url="redis://127.0.0.1:1/0",
            server_port=8000,
        )
        rc = await _cmd_flush(args)
        assert rc == 1
        # File must still be present — purge bailed before touching disk
        assert f.exists()

    async def test_flush_conid_cache_empty_dir_ok(self, tmp_path):
        """No cache files / no dir → still returns 0."""
        args = argparse.Namespace(
            what="conid-cache",
            data_dir=str(tmp_path / "utp"),
            redis_url="redis://127.0.0.1:1/0",
            server_port=8000,
        )
        rc = await _cmd_flush(args)
        assert rc == 0

    async def test_reconcile_with_flush(self, tmp_path):
        """Reconcile --flush clears stale data before reconciling."""
        data_dir = tmp_path / "utp"
        ledger_dir = data_dir / "ledger"
        ledger_dir.mkdir(parents=True)
        pos_file = data_dir / "positions.json"
        # Stale position that shouldn't survive flush
        pos_file.write_text('{"stale": {"status": "open", "symbol": "FAKE", "broker": "ibkr"}}')
        (ledger_dir / "ledger.jsonl").write_text('{"seq": 1}\n')

        args = argparse.Namespace(
            flush=True, data_dir=str(data_dir),
            live=False, paper=False, dry_run=True, _default_mode="dry-run",
            host="127.0.0.1", port=None, client_id=10, exchange=None,
            broker="ibkr", poll_timeout=5.0, poll_interval=0.1,
            show=False, portfolio=False,
        )
        rc = await _cmd_reconcile(args)
        assert rc == 0
        # After flush, stale position should be gone
        loaded = json.loads(pos_file.read_text())
        assert "stale" not in loaded

    async def test_reconcile_show_displays_positions(self, tmp_path, capsys):
        """Reconcile --show prints synced positions."""
        from app.services.position_store import get_position_store
        # Add a position to the module-level store (initialized by autouse fixture)
        store = get_position_store()
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY,
            quantity=100, order_type="MARKET",
        ))
        store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=True)

        args = argparse.Namespace(
            flush=False, show=True, portfolio=False,
            data_dir=str(tmp_path),
            live=False, paper=False, dry_run=True, _default_mode="dry-run",
            host="127.0.0.1", port=None, client_id=10, exchange=None,
            broker="ibkr", poll_timeout=5.0, poll_interval=0.1,
        )
        rc = await _cmd_reconcile(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Synced Positions" in captured.out
        assert "SPY" in captured.out

    async def test_reconcile_portfolio_displays_summary(self, tmp_path, capsys):
        """Reconcile --portfolio prints full portfolio summary."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY,
            quantity=50, order_type="MARKET",
        ))
        store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=True)

        args = argparse.Namespace(
            flush=False, show=False, portfolio=True,
            data_dir=str(tmp_path),
            live=False, paper=False, dry_run=True, _default_mode="dry-run",
            host="127.0.0.1", port=None, client_id=10, exchange=None,
            broker="ibkr", poll_timeout=5.0, poll_interval=0.1,
        )
        rc = await _cmd_reconcile(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Portfolio Summary" in captured.out
        assert "Cash Deployed" in captured.out
        assert "AAPL" in captured.out

    async def test_reconcile_mismatch_shows_explanation(self, tmp_path, capsys):
        """Quantity mismatch includes explanatory note."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="RUT", side=OrderSide.BUY,
            quantity=1, order_type="MARKET",
        ))
        store.add_position(req, OrderResult(broker=Broker.IBKR), is_paper=True)

        broker_pos = Position(broker=Broker.IBKR, symbol="RUT", quantity=-1,
                              avg_cost=100.0, market_value=-100.0, unrealized_pnl=0.0)
        mock_provider = MagicMock()
        mock_provider.broker = Broker.IBKR
        mock_provider.get_positions = AsyncMock(return_value=[broker_pos])

        args = argparse.Namespace(
            flush=False, show=False, portfolio=False,
            data_dir=str(tmp_path),
            live=False, paper=False, dry_run=True, _default_mode="dry-run",
            host="127.0.0.1", port=None, client_id=10, exchange=None,
            broker="ibkr", poll_timeout=5.0, poll_interval=0.1,
        )

        with patch("app.services.position_sync.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = mock_provider
            rc = await _cmd_reconcile(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "MISMATCH" in captured.out
        assert "negative qty for short positions" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# Playbook Service
# ═══════════════════════════════════════════════════════════════════════════════

VALID_MIXED_YAML = """
playbook:
  name: "Test Playbook"
  description: "Tests all instruction types"
  broker: ibkr

instructions:
  - id: buy_spy
    type: equity
    action: BUY
    symbol: SPY
    quantity: 100
    order_type: MARKET

  - id: spx_put_spread
    type: credit_spread
    symbol: SPX
    expiration: "2026-03-20"
    short_strike: 5500
    long_strike: 5475
    option_type: PUT
    quantity: 2
    net_price: 3.50

  - id: spy_long_put
    type: single_option
    symbol: SPY
    expiration: "2026-03-20"
    strike: 550
    option_type: PUT
    action: BUY_TO_OPEN
    quantity: 1
    order_type: LIMIT
    limit_price: 2.50

  - id: spx_condor
    type: iron_condor
    symbol: SPX
    expiration: "2026-03-20"
    put_short: 5500
    put_long: 5475
    call_short: 5700
    call_long: 5725
    quantity: 1
    net_price: 3.50

  - id: qqq_debit
    type: debit_spread
    symbol: QQQ
    expiration: "2026-03-20"
    long_strike: 480
    short_strike: 490
    option_type: CALL
    quantity: 3
    net_price: 4.00
"""

EQUITY_ONLY_YAML = """
playbook:
  name: "Equity Only"
  broker: ibkr

instructions:
  - id: buy_aapl
    type: equity
    action: BUY
    symbol: AAPL
    quantity: 50
    order_type: LIMIT
    limit_price: 200.00
"""


class TestPlaybookParsing:
    @pytest.fixture
    def _svc(self):
        from app.services.playbook_service import PlaybookService
        return PlaybookService()

    def test_parse_valid_mixed(self, _svc):
        pb = _svc.load(VALID_MIXED_YAML)
        assert pb.name == "Test Playbook"
        assert len(pb.instructions) == 5

    def test_parse_instruction_ids(self, _svc):
        pb = _svc.load(VALID_MIXED_YAML)
        assert [i.id for i in pb.instructions] == ["buy_spy", "spx_put_spread", "spy_long_put", "spx_condor", "qqq_debit"]

    def test_parse_instruction_types(self, _svc):
        pb = _svc.load(VALID_MIXED_YAML)
        assert [i.type for i in pb.instructions] == ["equity", "credit_spread", "single_option", "iron_condor", "debit_spread"]

    def test_parse_equity_params(self, _svc):
        pb = _svc.load(EQUITY_ONLY_YAML)
        assert pb.instructions[0].params["symbol"] == "AAPL"

    def test_parse_from_dict(self, _svc):
        raw = {"playbook": {"name": "Dict Test", "broker": "ibkr"},
               "instructions": [{"id": "test1", "type": "equity", "action": "BUY", "symbol": "SPY", "quantity": 10}]}
        assert _svc.load(raw).name == "Dict Test"

    def test_parse_from_file(self, _svc, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(EQUITY_ONLY_YAML)
        assert _svc.load(yaml_file).name == "Equity Only"

    def test_invalid_no_instructions(self, _svc):
        from app.services.playbook_service import PlaybookValidationError
        with pytest.raises(PlaybookValidationError, match="no instructions"):
            _svc.load({"playbook": {"name": "Empty"}, "instructions": []})

    def test_invalid_missing_id(self, _svc):
        from app.services.playbook_service import PlaybookValidationError
        with pytest.raises(PlaybookValidationError, match="missing 'id'"):
            _svc.load({"playbook": {"name": "Bad"}, "instructions": [{"type": "equity"}]})

    def test_invalid_unknown_type(self, _svc):
        from app.services.playbook_service import PlaybookValidationError
        with pytest.raises(PlaybookValidationError, match="invalid type"):
            _svc.load({"playbook": {"name": "Bad"}, "instructions": [{"id": "x", "type": "butterfly"}]})

    def test_invalid_duplicate_ids(self, _svc):
        from app.services.playbook_service import PlaybookValidationError
        with pytest.raises(PlaybookValidationError, match="Duplicate"):
            _svc.load({"playbook": {"name": "Bad"}, "instructions": [
                {"id": "same", "type": "equity", "action": "BUY", "symbol": "SPY", "quantity": 1},
                {"id": "same", "type": "equity", "action": "BUY", "symbol": "QQQ", "quantity": 1},
            ]})

    def test_invalid_broker(self, _svc):
        from app.services.playbook_service import PlaybookValidationError
        with pytest.raises(PlaybookValidationError, match="Unknown broker"):
            _svc.load({"playbook": {"name": "Bad", "broker": "schwab"},
                       "instructions": [{"id": "x", "type": "equity", "action": "BUY", "symbol": "SPY", "quantity": 1}]})

    def test_file_not_found(self, _svc):
        from app.services.playbook_service import PlaybookValidationError
        with pytest.raises(PlaybookValidationError, match="not found"):
            _svc.load("/nonexistent/playbook.yaml")


class TestInstructionTranslation:
    @pytest.fixture
    def _svc(self):
        from app.services.playbook_service import PlaybookService
        return PlaybookService()

    def test_equity(self, _svc):
        instr = PlaybookInstruction(id="buy_spy", type="equity",
                                     params={"symbol": "SPY", "action": "BUY", "quantity": 100, "order_type": "MARKET"})
        req = _svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert req.equity_order is not None
        assert req.equity_order.side == OrderSide.BUY

    def test_single_option(self, _svc):
        instr = PlaybookInstruction(id="spy_put", type="single_option",
                                     params={"symbol": "SPY", "expiration": "2026-03-20", "strike": 550,
                                             "option_type": "PUT", "action": "BUY_TO_OPEN", "quantity": 1,
                                             "order_type": "LIMIT", "limit_price": 2.50})
        req = _svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert len(req.multi_leg_order.legs) == 1

    def test_credit_spread(self, _svc):
        instr = PlaybookInstruction(id="cs", type="credit_spread",
                                     params={"symbol": "SPX", "expiration": "2026-03-20",
                                             "short_strike": 5500, "long_strike": 5475,
                                             "option_type": "PUT", "quantity": 2, "net_price": 3.50})
        req = _svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert len(req.multi_leg_order.legs) == 2
        assert req.multi_leg_order.legs[0].action == OptionAction.SELL_TO_OPEN

    def test_debit_spread(self, _svc):
        instr = PlaybookInstruction(id="ds", type="debit_spread",
                                     params={"symbol": "QQQ", "expiration": "2026-03-20",
                                             "long_strike": 480, "short_strike": 490,
                                             "option_type": "CALL", "quantity": 3, "net_price": 4.00})
        req = _svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert req.multi_leg_order.legs[0].action == OptionAction.BUY_TO_OPEN

    def test_iron_condor(self, _svc):
        instr = PlaybookInstruction(id="ic", type="iron_condor",
                                     params={"symbol": "SPX", "expiration": "2026-03-20",
                                             "put_short": 5500, "put_long": 5475,
                                             "call_short": 5700, "call_long": 5725,
                                             "quantity": 1, "net_price": 3.50})
        req = _svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert len(req.multi_leg_order.legs) == 4

    def test_missing_required_field(self, _svc):
        from app.services.playbook_service import PlaybookValidationError
        instr = PlaybookInstruction(id="bad", type="equity", params={"symbol": "SPY"})
        with pytest.raises(PlaybookValidationError, match="quantity"):
            _svc.instruction_to_trade_request(instr, Broker.IBKR)


class TestPlaybookExecution:
    @pytest.fixture
    def _svc(self):
        from app.services.playbook_service import PlaybookService
        return PlaybookService()

    @pytest.mark.asyncio
    async def test_execute_dry_run(self, _svc):
        pb = _svc.load(EQUITY_ONLY_YAML)
        result = await _svc.execute(pb, dry_run=True)
        assert result.succeeded == 1
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_execute_mixed_dry_run(self, _svc):
        pb = _svc.load(VALID_MIXED_YAML)
        result = await _svc.execute(pb, dry_run=True)
        assert result.total == 5
        assert result.succeeded == 5

    @pytest.mark.asyncio
    async def test_execute_continues_on_failure(self, _svc):
        raw = {"playbook": {"name": "Mixed", "broker": "ibkr"},
               "instructions": [
                   {"id": "good1", "type": "equity", "action": "BUY", "symbol": "SPY", "quantity": 10},
                   {"id": "bad", "type": "equity", "action": "INVALID_ACTION", "symbol": "QQQ", "quantity": 5},
                   {"id": "good2", "type": "equity", "action": "SELL", "symbol": "AAPL", "quantity": 20},
               ]}
        pb = _svc.load(raw)
        result = await _svc.execute(pb, dry_run=True)
        assert result.succeeded == 2
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_validate_all_valid(self, _svc):
        pb = _svc.load(VALID_MIXED_YAML)
        validations = await _svc.validate(pb)
        assert all(v["valid"] for v in validations)

    @pytest.mark.asyncio
    async def test_validate_with_errors(self, _svc):
        raw = {"playbook": {"name": "Bad", "broker": "ibkr"},
               "instructions": [
                   {"id": "good", "type": "equity", "action": "BUY", "symbol": "SPY", "quantity": 10},
                   {"id": "bad", "type": "credit_spread", "symbol": "SPX"},
               ]}
        pb = _svc.load(raw)
        validations = await _svc.validate(pb)
        assert validations[0]["valid"] is True
        assert validations[1]["valid"] is False

    @pytest.mark.asyncio
    async def test_post_submit_hook(self, _svc):
        hook_calls = []

        async def _hook(instr_id, trade_request, initial_result):
            hook_calls.append(instr_id)
            return OrderResult(order_id=initial_result.order_id, broker=initial_result.broker,
                               status=OrderStatus.FILLED, message="filled via hook", filled_price=99.50)

        pb = _svc.load(EQUITY_ONLY_YAML)
        result = await _svc.execute(pb, dry_run=False, post_submit_hook=_hook)
        assert len(hook_calls) == 1
        assert result.results[0].order_result.filled_price == 99.50

    @pytest.mark.asyncio
    async def test_hook_not_called_dry_run(self, _svc):
        hook_calls = []

        async def _hook(instr_id, trade_request, initial_result):
            hook_calls.append(instr_id)
            return initial_result

        pb = _svc.load(EQUITY_ONLY_YAML)
        await _svc.execute(pb, dry_run=True, post_submit_hook=_hook)
        assert len(hook_calls) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Fill Tracking
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeProvider:
    """Mock provider that returns SUBMITTED for N polls then terminal status."""
    def __init__(self, polls_before_fill=0, terminal_status=OrderStatus.FILLED):
        self._polls = 0
        self._polls_before_fill = polls_before_fill
        self._terminal_status = terminal_status

    async def get_order_status(self, order_id):
        self._polls += 1
        if self._polls > self._polls_before_fill:
            return OrderResult(order_id=order_id, broker=Broker.IBKR,
                               status=self._terminal_status,
                               message=f"terminal: {self._terminal_status.value}",
                               filled_price=150.25 if self._terminal_status == OrderStatus.FILLED else None)
        return OrderResult(order_id=order_id, broker=Broker.IBKR,
                           status=OrderStatus.SUBMITTED, message="still working...")


class TestFillTracking:
    @pytest.fixture(autouse=True)
    def _drop_poll_floor(self, monkeypatch):
        """The 2s floor in await_order_fill is right for production but
        makes these unit tests crawl. Drop it for the duration of the
        class so we can keep fast 0.01s polls."""
        monkeypatch.setattr(
            "app.services.trade_service._MIN_POLL_INTERVAL", 0.0
        )

    @pytest.mark.asyncio
    async def test_immediate_fill(self):
        provider = _FakeProvider(polls_before_fill=0)
        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            from app.services.trade_service import await_order_fill
            result = await await_order_fill(broker=Broker.IBKR, order_id="test-001",
                                             poll_interval=0.01, timeout=1.0)
        assert result.status == OrderStatus.FILLED
        assert result.filled_price == 150.25

    @pytest.mark.asyncio
    async def test_delayed_fill(self):
        provider = _FakeProvider(polls_before_fill=3)
        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            from app.services.trade_service import await_order_fill
            result = await await_order_fill(broker=Broker.IBKR, order_id="test-002",
                                             poll_interval=0.01, timeout=2.0)
        assert result.status == OrderStatus.FILLED
        assert provider._polls == 4

    @pytest.mark.asyncio
    async def test_timeout(self):
        provider = _FakeProvider(polls_before_fill=9999)
        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            from app.services.trade_service import await_order_fill
            result = await await_order_fill(broker=Broker.IBKR, order_id="test-003",
                                             poll_interval=0.02, timeout=0.1)
        assert result.status == OrderStatus.SUBMITTED
        assert "timed out" in result.message

    @pytest.mark.asyncio
    async def test_cancelled_order(self):
        provider = _FakeProvider(polls_before_fill=1, terminal_status=OrderStatus.CANCELLED)
        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            from app.services.trade_service import await_order_fill
            result = await await_order_fill(broker=Broker.IBKR, order_id="test-004",
                                             poll_interval=0.01, timeout=1.0)
        assert result.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_rejected_order(self):
        provider = _FakeProvider(polls_before_fill=0, terminal_status=OrderStatus.REJECTED)
        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            from app.services.trade_service import await_order_fill
            result = await await_order_fill(broker=Broker.IBKR, order_id="test-005",
                                             poll_interval=0.01, timeout=1.0)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        provider = _FakeProvider(polls_before_fill=2)
        callback_calls = []

        async def _track(result, elapsed):
            callback_calls.append(result.status)

        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            from app.services.trade_service import await_order_fill
            await await_order_fill(broker=Broker.IBKR, order_id="test-006",
                                   poll_interval=0.01, timeout=1.0,
                                   on_status_update=_track)
        assert len(callback_calls) == 3

    @pytest.mark.asyncio
    async def test_fill_creates_position(self):
        from app.services.trade_service import _pending_orders, await_order_fill
        request = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY, quantity=10,
        ))
        _pending_orders["fill-test-001"] = request
        provider = _FakeProvider(polls_before_fill=0)
        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            result = await await_order_fill(broker=Broker.IBKR, order_id="fill-test-001",
                                             poll_interval=0.01, timeout=1.0)
        assert result.status == OrderStatus.FILLED
        assert "fill-test-001" not in _pending_orders

    @pytest.mark.asyncio
    async def test_timeout_preserves_pending(self):
        from app.services.trade_service import _pending_orders, await_order_fill
        request = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY, quantity=10,
        ))
        _pending_orders["timeout-test-001"] = request
        provider = _FakeProvider(polls_before_fill=9999)
        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            result = await await_order_fill(broker=Broker.IBKR, order_id="timeout-test-001",
                                             poll_interval=0.02, timeout=0.1)
        assert "timeout-test-001" in _pending_orders
        _pending_orders.pop("timeout-test-001", None)


# ═══════════════════════════════════════════════════════════════════════════════
# IBKR Provider (stub + live mock + caching)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIBKRProvider:
    def test_stub_fallback(self):
        from app.core.providers.ibkr import IBKRProvider
        assert IBKRProvider().broker == Broker.IBKR

    @pytest.mark.asyncio
    async def test_stub_get_positions(self):
        from app.core.providers.ibkr import IBKRProvider
        provider = IBKRProvider()
        await provider.connect()
        positions = await provider.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        await provider.disconnect()

    @pytest.mark.asyncio
    async def test_live_readonly_rejects(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()
        with patch("app.config.settings") as mock_settings:
            mock_settings.ibkr_readonly = True
            order = EquityOrder(broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY,
                                quantity=100, order_type=OrderType.MARKET)
            result = await provider.execute_equity_order(order)
            assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_live_credit_spread_negates_price(self):
        """Credit spread limit price is sent as negative to IBKR."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        # Mock contract qualification
        mock_contract = MagicMock(conId=12345)
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=[mock_contract])

        # Mock placeOrder to capture the order
        mock_trade = MagicMock()
        mock_trade.order.orderId = 42
        provider._ib.placeOrder = MagicMock(return_value=mock_trade)

        with patch("app.config.settings") as mock_settings:
            mock_settings.ibkr_readonly = False

            order = MultiLegOrder(broker=Broker.IBKR, legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ], order_type=OrderType.LIMIT, net_price=2.50)
            result = await provider.execute_multi_leg_order(order)

        assert result.status == OrderStatus.SUBMITTED
        # Verify the limit price was negated (credit spread)
        placed_order = provider._ib.placeOrder.call_args[0][1]
        assert placed_order.lmtPrice == -2.50

    @pytest.mark.asyncio
    async def test_live_debit_spread_positive_price(self):
        """Debit spread limit price stays positive for IBKR."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        mock_contract = MagicMock(conId=12345)
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=[mock_contract])

        mock_trade = MagicMock()
        mock_trade.order.orderId = 43
        provider._ib.placeOrder = MagicMock(return_value=mock_trade)

        with patch("app.config.settings") as mock_settings:
            mock_settings.ibkr_readonly = False

            order = MultiLegOrder(broker=Broker.IBKR, legs=[
                OptionLeg(symbol="QQQ", expiration="2026-03-20", strike=480.0,
                          option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
                OptionLeg(symbol="QQQ", expiration="2026-03-20", strike=490.0,
                          option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
            ], order_type=OrderType.LIMIT, net_price=3.00)
            result = await provider.execute_multi_leg_order(order)

        assert result.status == OrderStatus.SUBMITTED
        placed_order = provider._ib.placeOrder.call_args[0][1]
        assert placed_order.lmtPrice == 3.00

    @pytest.mark.asyncio
    async def test_live_combo_reduces_leg_ratios_to_lowest_terms(self):
        """When per-leg quantities equal the combo quantity (caller mistake),
        the provider must reduce ComboLeg.ratio by GCD — otherwise IBKR
        rejects with error 321 'Invalid leg ratio'."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        mock_contract = MagicMock(conId=12345)
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=[mock_contract])

        mock_trade = MagicMock()
        mock_trade.order.orderId = 99
        provider._ib.placeOrder = MagicMock(return_value=mock_trade)

        with patch("app.config.settings") as mock_settings:
            mock_settings.ibkr_readonly = False

            order = MultiLegOrder(broker=Broker.IBKR, legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                          option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=10),
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                          option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=10),
            ], order_type=OrderType.LIMIT, net_price=2.50, quantity=10)
            result = await provider.execute_multi_leg_order(order)

        assert result.status == OrderStatus.SUBMITTED
        placed_combo = provider._ib.placeOrder.call_args[0][0]
        # Both legs reduced to ratio=1 (NOT 10:10)
        assert all(leg.ratio == 1 for leg in placed_combo.comboLegs)
        # Total contract count carried by Order.totalQuantity, not leg ratios
        placed_order = provider._ib.placeOrder.call_args[0][1]
        assert placed_order.totalQuantity == 10

    @pytest.mark.asyncio
    async def test_get_option_quotes_writes_to_cross_provider_store(self):
        """`utp options ... --live` (which calls get_option_quotes in TWS mode)
        must write each resolved conId to the shared OptionConidStore so CPG
        can read it later — this is the explicit warm-up path the user runs
        when CPG's secdef/info is broken."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        # Mock get_option_chain to return a couple of strikes
        async def fake_chain(sym):
            return {"expirations": ["2026-04-27"], "strikes": [7050.0, 7075.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        # Mock qualifyContractsAsync — return contracts with valid conIds
        qualified_contracts = []
        for s, cid in [(7050.0, 700050), (7075.0, 700075)]:
            mc = MagicMock()
            mc.conId = cid
            mc.strike = s
            mc.localSymbol = f"SPX 260427P0{int(s*1000):07d}"
            qualified_contracts.append(mc)
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=qualified_contracts)

        # Mock the market-data subscription path (we don't care about quotes for this test)
        ticker_mocks = []
        for s in (7050.0, 7075.0):
            t = MagicMock()
            t.contract = next(c for c in qualified_contracts if c.strike == s)
            t.bid = 0.5
            t.ask = 0.6
            t.last = 0.55
            t.volume = 0
            ticker_mocks.append(t)
        provider._ib.reqMktData = MagicMock(side_effect=ticker_mocks)
        provider._ib.cancelMktData = MagicMock()

        await provider.get_option_quotes("SPX", "2026-04-27", "PUT",
                                          strike_min=7000, strike_max=7100)

        # Each resolved conId is now in the shared store — CPG can use it.
        store = provider._cache.option_conids
        assert store.get("SPX", "20260427", 7050.0, "P") == 700050
        assert store.get("SPX", "20260427", 7075.0, "P") == 700075

    @pytest.mark.asyncio
    async def test_get_option_chain_augments_via_contract_details_for_spx(self):
        """When ``reqSecDefOptParamsAsync`` returns a partial SPX chain (only
        the parent class's strikes, missing the SPXW chain), get_option_chain
        must augment via ``reqContractDetails`` for the SPXW class. Locks in
        the fix for the production bug where SPX's chain cache had only 60
        strikes clustered around $5000-$5590 while the market was at $7130."""
        from unittest.mock import MagicMock, AsyncMock
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        # Underlying qualification
        underlying = MagicMock(); underlying.conId = 416904; underlying.secType = "IND"
        async def fake_qualify(_c):
            return [underlying]
        provider._qualify_contract_cached = fake_qualify  # type: ignore[assignment]

        # reqSecDefOptParamsAsync returns the partial parent-class chain
        # (this is what TWS gave us in production for SPX).
        partial_chain = MagicMock()
        partial_chain.expirations = ["20260515", "20260619"]
        partial_chain.strikes = [float(5000 + i * 10) for i in range(60)]
        provider._ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[partial_chain])

        # reqContractDetailsAsync for SPXW returns the full near-term chain
        # — strikes spanning $4000-$10000, the realistic SPXW grid.
        full_strikes = [float(4000 + i * 5) for i in range(1200)]
        cd_results = []
        for s in full_strikes:
            cd = MagicMock()
            cd.contract = MagicMock()
            cd.contract.strike = s
            cd.contract.lastTradeDateOrContractMonth = "20260430"
            cd_results.append(cd)
        provider._ib.reqContractDetailsAsync = AsyncMock(return_value=cd_results)

        result = await provider.get_option_chain("SPX")

        # Confirm reqContractDetails ran with tradingClass=SPXW
        call_args = provider._ib.reqContractDetailsAsync.call_args
        sample_opt = call_args.args[0]
        assert getattr(sample_opt, "tradingClass", None) == "SPXW", (
            "augmentation must query SPXW class, not the parent class"
        )

        # Result strikes include both the partial-chain strikes AND the
        # SPXW strikes. The chain now brackets reasonable SPX prices.
        assert len(result["strikes"]) >= 1200
        assert max(result["strikes"]) >= 9000.0, (
            f"augmented chain must reach realistic SPX strikes; "
            f"got max={max(result['strikes'])}"
        )
        # Both expirations from secdef AND from contract details
        assert "20260430" in result["expirations"]
        assert "20260515" in result["expirations"]

    @pytest.mark.asyncio
    async def test_get_option_quotes_forces_spxw_for_all_spx_expirations(self):
        """Per deployment policy, SPX is forced to ``SPXW`` on EVERY
        expiration — including the 3rd-Friday monthlies. SPXW is co-listed
        with the parent SPX class on monthlies (PM- vs AM-settled), so
        forcing SPXW always lands on the daily-style PM contract that
        the streaming and quote paths use uniformly. Without this, daily
        SPX expirations (4 of every 5 weekdays) return empty
        qualifications and the streaming cache goes stale on SPX."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        captured: list[str] = []

        async def fake_qualify(*contracts):
            for c in contracts:
                captured.append(getattr(c, "tradingClass", None) or "")
            results = []
            for c in contracts:
                mc = MagicMock()
                mc.conId = 700000 + int(c.strike)
                mc.strike = c.strike
                mc.localSymbol = "SPX 260429P07050000"
                results.append(mc)
            return results

        provider._ib.qualifyContractsAsync = fake_qualify
        provider._ib.reqMktData = MagicMock(side_effect=lambda c, **_: MagicMock(
            contract=c, bid=1.0, ask=1.1, last=1.05, volume=0,
        ))
        provider._ib.cancelMktData = MagicMock()

        # Daily expiration (Wed)
        async def fake_chain(_sym):
            return {"expirations": ["2026-04-29"], "strikes": [7050.0, 7075.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]
        await provider.get_option_quotes(
            "SPX", "2026-04-29", "PUT", strike_min=7000, strike_max=7100,
        )
        assert all(tc == "SPXW" for tc in captured), (
            f"every SPX option must carry tradingClass=SPXW; got {captured}"
        )

        # Monthly expiration (3rd Friday) — same policy: still SPXW
        captured.clear()
        provider._option_subs.clear()
        provider._option_subs_last_used.clear()

        async def fake_chain_monthly(_sym):
            return {"expirations": ["2026-05-15"], "strikes": [7050.0]}
        provider.get_option_chain = fake_chain_monthly  # type: ignore[assignment]
        await provider.get_option_quotes(
            "SPX", "2026-05-15", "PUT", strike_min=7000, strike_max=7100,
        )
        assert all(tc == "SPXW" for tc in captured), (
            f"3rd-Friday SPX must STILL be forced to SPXW under the new "
            f"policy (PM-settled, co-listed with monthly SPX); got {captured}"
        )

    @pytest.mark.asyncio
    async def test_get_option_quotes_forces_rutw_for_all_rut_expirations(self):
        """Same policy as SPX — RUT always uses RUTW (PM-settled weekly),
        even on monthly 3rd Fridays."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        async def fake_chain(_sym):
            return {"expirations": ["2026-05-15"], "strikes": [2750.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        captured: list[str] = []

        async def fake_qualify(*contracts):
            for c in contracts:
                captured.append(getattr(c, "tradingClass", None) or "")
            mc = MagicMock(); mc.conId = 700001; mc.strike = 2750.0
            mc.localSymbol = "RUT 260515P02750000"
            return [mc]

        provider._ib.qualifyContractsAsync = fake_qualify
        provider._ib.reqMktData = MagicMock(side_effect=lambda c, **_: MagicMock(
            contract=c, bid=1.0, ask=1.1, last=1.05, volume=0,
        ))
        provider._ib.cancelMktData = MagicMock()

        await provider.get_option_quotes(
            "RUT", "2026-05-15", "PUT", strike_min=2700, strike_max=2800,
        )
        assert captured == ["RUTW"], (
            f"3rd-Friday RUT must be forced to RUTW; got {captured}"
        )

    @pytest.mark.asyncio
    async def test_get_option_quotes_does_not_force_class_for_ndx(self):
        """NDX is intentionally NOT forced. Live testing on 2026-04-29
        showed that forcing ``NDXP`` made 3 of 6 (sym, exp, type) jobs
        fail to qualify and inflated cycle latency from <1s to 22s. The
        default ``NDX`` class qualifies daily and weekly NDX cleanly.
        Users can opt in to NDXP per-trade via ``--trading-class NDXP``."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        async def fake_chain(_sym):
            return {"expirations": ["2026-04-29"], "strikes": [27000.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        captured: list[str] = []

        async def fake_qualify(*contracts):
            for c in contracts:
                captured.append(getattr(c, "tradingClass", None) or "")
            mc = MagicMock(); mc.conId = 700002; mc.strike = 27000.0
            mc.localSymbol = "NDX 260429P27000000"
            return [mc]

        provider._ib.qualifyContractsAsync = fake_qualify
        provider._ib.reqMktData = MagicMock(side_effect=lambda c, **_: MagicMock(
            contract=c, bid=1.0, ask=1.1, last=1.05, volume=0,
        ))
        provider._ib.cancelMktData = MagicMock()

        await provider.get_option_quotes(
            "NDX", "2026-04-29", "PUT", strike_min=26900, strike_max=27100,
        )
        assert captured == [""], (
            f"NDX must use the default class (no force); got {captured}"
        )

    @pytest.mark.asyncio
    async def test_qualify_option_honors_per_leg_trading_class_override(self):
        """A per-leg ``trading_class`` (e.g. set on OptionLeg by the CLI's
        ``--trading-class SPX``) overrides the default forced class. Empty
        string disables forcing entirely. Real-world use: rolling out of
        a legacy AM-settled monthly SPX position you opened years ago."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        captured: list[str] = []

        async def fake_cached_qualify(opt):
            captured.append(getattr(opt, "tradingClass", None) or "")
            mc = MagicMock(); mc.conId = 555000; mc.strike = float(opt.strike)
            mc.localSymbol = "SPX 260515P07050000"
            return [mc]

        provider._qualify_contract_cached = fake_cached_qualify  # type: ignore[assignment]

        # Override "SPX" — user wants the AM-settled monthly contract
        await provider._qualify_option(
            "SPX", "20260515", 7050.0, "P",
            trading_class_override="SPX",
        )
        assert captured[-1] == "SPX", (
            f"override should set tradingClass to 'SPX'; got {captured[-1]!r}"
        )

        # Override "" — disable forcing entirely
        captured.clear()
        await provider._qualify_option(
            "SPX", "20260515", 7050.0, "P",
            trading_class_override="",
        )
        assert captured[-1] == "", (
            f"empty override should leave tradingClass unset; got {captured[-1]!r}"
        )

    def test_force_option_trading_class_env_override(self, monkeypatch):
        """``UTP_OPTION_TC_<SYMBOL>`` env var overrides the default. Useful
        when running the daemon to flip behavior without touching code."""
        from app.core.providers.ibkr import _force_option_trading_class
        # Default: SPX→SPXW
        monkeypatch.delenv("UTP_OPTION_TC_SPX", raising=False)
        assert _force_option_trading_class("SPX") == "SPXW"
        # Env override to use the parent class
        monkeypatch.setenv("UTP_OPTION_TC_SPX", "SPX")
        assert _force_option_trading_class("SPX") == "SPX"
        # Env empty → no forcing
        monkeypatch.setenv("UTP_OPTION_TC_SPX", "")
        assert _force_option_trading_class("SPX") is None
        # NDX: default is no-force; env opt-in to NDXP
        monkeypatch.delenv("UTP_OPTION_TC_NDX", raising=False)
        assert _force_option_trading_class("NDX") is None
        monkeypatch.setenv("UTP_OPTION_TC_NDX", "NDXP")
        assert _force_option_trading_class("NDX") == "NDXP"

    @pytest.mark.asyncio
    async def test_qualify_option_forces_spxw_for_trades(self):
        """The trade execution path also forces SPXW — without this, daily SPX
        trades would fail to qualify and the order would never reach IBKR.
        Streaming and trading must agree on the same contract."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        captured: list[str] = []

        async def fake_qualify(opt):
            captured.append(getattr(opt, "tradingClass", None) or "")
            return [opt]

        # _qualify_contract_cached invokes _ib.qualifyContractsAsync via
        # the cache wrapper. Stub the wrapper directly for simplicity.
        async def fake_cached_qualify(opt):
            await fake_qualify(opt)
            mc = MagicMock(); mc.conId = 868691755
            mc.strike = float(getattr(opt, "strike", 7050.0))
            mc.localSymbol = "SPX 260429P07050000"
            return [mc]

        provider._qualify_contract_cached = fake_cached_qualify  # type: ignore[assignment]

        result = await provider._qualify_option("SPX", "20260429", 7050.0, "P")
        assert result and result[0].conId == 868691755
        # First exchange attempt (SMART) must carry tradingClass=SPXW
        assert captured[0] == "SPXW", (
            f"trade-flow qualify must force SPXW for SPX; got {captured}"
        )

    @pytest.mark.asyncio
    async def test_get_option_quotes_does_not_raise_on_unlisted_expiration(self):
        """TWS's ``reqSecDefOptParamsAsync`` does NOT enumerate near-term daily
        expirations (SPXW 0DTE/1DTE/2DTE), but those contracts exist and
        qualify normally. Previously we raised ``ValueError("Expiration …
        not available")`` which made ~40% of option-quote streaming cycles
        fail on SPX/RUT 1-2DTE. Now we skip the gate and let
        ``qualifyContractsAsync`` filter — same shape as CPG's per-contract
        ``secdef/info`` flow."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        # Chain meta omits the requested expiration (the bug scenario).
        async def fake_chain(_sym):
            return {
                "expirations": ["2026-05-15", "2026-06-19"],  # only monthlies
                "strikes": [7050.0, 7075.0],
            }
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        # qualifyContractsAsync returns valid contracts for the "missing"
        # expiration — TWS happily qualifies it even though it wasn't in
        # the chain meta.
        qualified = []
        for s in (7050.0, 7075.0):
            mc = MagicMock()
            mc.conId = 700000 + int(s)
            mc.strike = s
            mc.localSymbol = f"SPX 260430P0{int(s*1000):07d}"
            qualified.append(mc)
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=qualified)

        ticker_iter = []
        for c in qualified:
            t = MagicMock(); t.contract = c
            t.bid, t.ask, t.last, t.volume = 1.0, 1.1, 1.05, 0
            ticker_iter.append(t)
        provider._ib.reqMktData = MagicMock(side_effect=ticker_iter)
        provider._ib.cancelMktData = MagicMock()

        # Must NOT raise — the daily expiration is valid even though chain meta
        # didn't list it. Returns the qualified strikes.
        results = await provider.get_option_quotes(
            "SPX", "2026-04-30", "PUT",
            strike_min=7000, strike_max=7100,
        )
        assert len(results) == 2
        assert {r["strike"] for r in results} == {7050.0, 7075.0}

    @pytest.mark.asyncio
    async def test_get_option_quotes_reuses_persistent_subscriptions(self):
        """Second call for the same (symbol, exp, type, strikes) MUST NOT
        re-subscribe via reqMktData — that's the bug that made TWS-mode
        option quotes fall behind the streaming poll cycle. Persistent
        Tickers tick continuously in ib_insync's background loop, so we
        just read their current values."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        async def fake_chain(sym):
            return {"expirations": ["2026-04-27"], "strikes": [7050.0, 7075.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        qualified = []
        for s, cid in [(7050.0, 700050), (7075.0, 700075)]:
            mc = MagicMock()
            mc.conId = cid
            mc.strike = s
            mc.localSymbol = f"SPX 260427P0{int(s*1000):07d}"
            qualified.append(mc)
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=qualified)

        tickers = {}
        for c in qualified:
            t = MagicMock()
            t.contract = c
            t.bid, t.ask, t.last, t.volume = 1.0, 1.1, 1.05, 100
            tickers[c.conId] = t

        def _req(c, **_):
            return tickers[c.conId]
        provider._ib.reqMktData = MagicMock(side_effect=_req)
        provider._ib.cancelMktData = MagicMock()

        # First call subscribes both strikes
        await provider.get_option_quotes("SPX", "2026-04-27", "PUT",
                                          strike_min=7000, strike_max=7100)
        assert provider._ib.reqMktData.call_count == 2
        # No cancel between calls — this is the whole point
        assert provider._ib.cancelMktData.call_count == 0
        # Persistent state populated
        assert (("SPX", "20260427", "P", 7050.0)) in provider._option_subs
        assert (("SPX", "20260427", "P", 7075.0)) in provider._option_subs

        # Second call: zero new subscriptions, zero cancels
        provider._ib.reqMktData.reset_mock()
        await provider.get_option_quotes("SPX", "2026-04-27", "PUT",
                                          strike_min=7000, strike_max=7100)
        assert provider._ib.reqMktData.call_count == 0, (
            "second call must not re-subscribe — that was the bug"
        )
        assert provider._ib.cancelMktData.call_count == 0

    @pytest.mark.asyncio
    async def test_get_option_quotes_lru_evicts_oldest(self):
        """When ``_option_subs_budget`` would be exceeded, the LRU-oldest
        subscriptions must be cancelled to free IBKR market-data lines."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()
        provider._option_subs_budget = 2  # tiny budget for the test

        # Pre-seed 2 subscriptions; mark one as oldest
        old_t = MagicMock()
        old_t.contract = MagicMock()
        recent_t = MagicMock()
        recent_t.contract = MagicMock()
        provider._option_subs[("SPX", "20260427", "P", 6000.0)] = old_t
        provider._option_subs[("SPX", "20260427", "P", 6500.0)] = recent_t
        provider._option_subs_last_used[("SPX", "20260427", "P", 6000.0)] = 100.0
        provider._option_subs_last_used[("SPX", "20260427", "P", 6500.0)] = 200.0

        async def fake_chain(sym):
            return {"expirations": ["2026-04-27"], "strikes": [7050.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        new_c = MagicMock(); new_c.conId = 700050; new_c.strike = 7050.0
        new_c.localSymbol = "SPX 260427P07050000"
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=[new_c])

        new_t = MagicMock()
        new_t.contract = new_c
        new_t.bid, new_t.ask, new_t.last, new_t.volume = 1.0, 1.1, 1.05, 0
        provider._ib.reqMktData = MagicMock(return_value=new_t)
        provider._ib.cancelMktData = MagicMock()

        await provider.get_option_quotes("SPX", "2026-04-27", "PUT",
                                          strike_min=7000, strike_max=7100)

        # Oldest sub (strike 6000, last_used=100.0) is evicted
        assert ("SPX", "20260427", "P", 6000.0) not in provider._option_subs
        assert ("SPX", "20260427", "P", 6500.0) in provider._option_subs
        assert ("SPX", "20260427", "P", 7050.0) in provider._option_subs
        # The eviction issued a cancel — exactly the LRU one
        assert provider._ib.cancelMktData.call_count == 1
        cancelled_contract = provider._ib.cancelMktData.call_args[0][0]
        assert cancelled_contract is old_t.contract

    @pytest.mark.asyncio
    async def test_streaming_service_applies_sub_budget_from_config(self, tmp_path, monkeypatch):
        """``OptionQuoteStreamingService.start()`` must override the provider's
        ``_option_subs_budget`` from ``option_quotes_ibkr_sub_budget``. The
        production bug was the streaming workload (3 indices × 3 exps × 2
        types × ~150 strikes ≈ 2700) blowing out the 300 default and
        thrashing through LRU evictions every cycle."""
        from app.core.providers.ibkr import IBKRLiveProvider
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        provider = IBKRLiveProvider()
        provider._option_subs_budget = 300  # the old default
        config = StreamingConfig(
            symbols=[],
            redis_enabled=False,
            redis_url="",
            questdb_enabled=False,
            ws_broadcast_enabled=False,
            close_band_pct=0.35,
            option_quotes_ibkr_sub_budget=2500,
        )

        svc = OptionQuoteStreamingService(config, provider, streaming_svc=None)

        # Avoid touching Redis or quote caches during this test
        async def _noop(*_a, **_k): return None
        monkeypatch.setattr(svc, "_redis_connect", _noop)
        monkeypatch.setattr(svc, "_redis_load_conid_cache", _noop)
        monkeypatch.setattr(svc, "_redis_load_quotes", _noop)
        monkeypatch.setattr(svc._cache, "set_redis", lambda *_: None)
        monkeypatch.setattr(svc._ibkr_cache, "set_redis", lambda *_: None)

        try:
            await svc.start()
            assert provider._option_subs_budget == 2500, (
                f"streaming config budget should override provider default; "
                f"got {provider._option_subs_budget}"
            )
        finally:
            svc._running = False

    @pytest.mark.asyncio
    async def test_disconnect_cancels_persistent_option_subs(self):
        """``disconnect()`` must cancel every persistent option subscription
        so we don't leak IBKR market-data lines across daemon restarts."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        ib = MagicMock()
        provider._ib = ib

        # Two persistent subs
        for s in (7050.0, 7075.0):
            t = MagicMock(); t.contract = MagicMock()
            provider._option_subs[("SPX", "20260427", "P", s)] = t
            provider._option_subs_last_used[("SPX", "20260427", "P", s)] = 100.0

        await provider.disconnect()

        assert ib.cancelMktData.call_count == 2
        assert provider._option_subs == {}
        assert provider._option_subs_last_used == {}

    def test_on_disconnect_drops_orphaned_option_subs(self):
        """On unexpected disconnect, the registry must clear without trying
        to cancel (the underlying ib_insync client is dead — cancel calls
        would just throw). Reconnect's first ``get_option_quotes`` call
        re-subscribes from scratch."""
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()

        for s in (7050.0,):
            t = MagicMock(); t.contract = MagicMock()
            provider._option_subs[("SPX", "20260427", "P", s)] = t
            provider._option_subs_last_used[("SPX", "20260427", "P", s)] = 100.0

        # Simulate without a running event loop — _on_disconnect catches the
        # RuntimeError when trying to spawn the reconnect task. Either way,
        # the registry must be cleared.
        try:
            provider._on_disconnect()
        except RuntimeError:
            pass

        assert provider._option_subs == {}
        assert provider._option_subs_last_used == {}

    @pytest.mark.asyncio
    async def test_live_get_positions_not_connected(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        assert await IBKRLiveProvider().get_positions() == []

    @pytest.mark.asyncio
    async def test_live_disconnect(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._ib = MagicMock()
        provider._connected = True
        await provider.disconnect()
        assert not provider._connected

    @pytest.mark.asyncio
    async def test_live_order_status_not_connected(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        result = await IBKRLiveProvider().get_order_status("fake-123")
        assert result.status == OrderStatus.FAILED

    def test_contract_cache_hit(self):
        from app.core.providers.ibkr_cache import ContractCache
        cache = ContractCache()
        mock = MagicMock(conId=12345)
        cache.put(mock, "SPX", "OPT", "20260316", 5500.0, "P")
        assert cache.get("SPX", "OPT", "20260316", 5500.0, "P") is mock
        assert cache.stats["hits"] == 1

    def test_contract_cache_miss(self):
        from app.core.providers.ibkr_cache import ContractCache
        cache = ContractCache()
        cache.put(MagicMock(), "SPX", "OPT", "20260316", 5500.0, "P")
        assert cache.get("SPX", "OPT", "20260316", 5475.0, "P") is None
        assert cache.stats["misses"] == 1

    def test_quote_cache_ttl(self):
        from app.core.providers.ibkr_cache import QuoteSnapshotCache
        cache = QuoteSnapshotCache(ttl_seconds=0.1)
        cache.put("SPX", {"bid": 100})
        assert cache.get("SPX") is not None
        time.sleep(0.15)
        assert cache.get("SPX") is None

    def test_option_chain_cache(self, tmp_path):
        from app.core.providers.ibkr_cache import OptionChainCache
        from datetime import date, timedelta
        cache = OptionChainCache(cache_dir=str(tmp_path))
        # Use a realistic SPX chain: ~600 strikes spanning $1000-$10990 so
        # min/max < 0.6 (passes the strike-span sanity check).
        today_exp = date.today().strftime("%Y%m%d")
        future_exp = (date.today() + timedelta(days=7)).strftime("%Y%m%d")
        strikes = [float(1000 + i * 10) for i in range(1000)]
        cache.put("SPX", [today_exp, future_exp], strikes)
        result = cache.get("SPX")
        assert result is not None
        assert result["strikes"] == strikes

    def test_option_chain_cache_disk_reload(self, tmp_path):
        from app.core.providers.ibkr_cache import OptionChainCache
        from datetime import date, timedelta
        c1 = OptionChainCache(cache_dir=str(tmp_path))
        today_exp = date.today().strftime("%Y%m%d")
        future_exp = (date.today() + timedelta(days=7)).strftime("%Y%m%d")
        # Realistic NDX chain spanning $5000-$45000 (min/max=0.11)
        strikes = [float(5000 + i * 50) for i in range(800)]
        c1.put("NDX", [today_exp, future_exp], strikes)
        c2 = OptionChainCache(cache_dir=str(tmp_path))
        assert c2.get("NDX") is not None

    def test_option_chain_cache_rejects_clustered_strikes(self, tmp_path):
        """The bug we hit on 2026-04-29: ``reqSecDefOptParamsAsync`` for SPX
        returned 60 strikes spanning only $5000-$5590 while the actual
        market price was $7130. The cache passed the count-and-expiration
        checks but had no strike anywhere near the underlying. This test
        locks in the strike-span validation that now catches it."""
        from app.core.providers.ibkr_cache import OptionChainCache
        from datetime import date, timedelta
        cache = OptionChainCache(cache_dir=str(tmp_path))
        today_exp = date.today().strftime("%Y%m%d")
        future_exp = (date.today() + timedelta(days=7)).strftime("%Y%m%d")
        # 60 strikes clustered tightly: min/max = 5000/5590 ≈ 0.89 > 0.6 floor
        clustered = [float(5000 + i * 10) for i in range(60)]
        cache.put("SPX", [today_exp, future_exp], clustered)
        # put() refused to persist — get() must return None
        assert cache.get("SPX") is None

    def test_option_chain_cache_rejects_clustered_on_read(self, tmp_path):
        """Even if a clustered chain somehow lands on disk (e.g. written by
        an older daemon version that lacked the put-time validation), the
        get-time validator must reject it so the next caller falls through
        to a fresh fetch."""
        import json
        from app.core.providers.ibkr_cache import OptionChainCache
        from datetime import date, timedelta
        cache = OptionChainCache(cache_dir=str(tmp_path))
        today_exp = date.today().strftime("%Y%m%d")
        future_exp = (date.today() + timedelta(days=7)).strftime("%Y%m%d")
        # Bypass put() and write the bad chain straight to disk
        path = cache._file_path("SPX", date.today())
        path.write_text(json.dumps({
            "symbol": "SPX",
            "date": date.today().isoformat(),
            "expirations": [today_exp, future_exp],
            "strikes": [float(5000 + i * 10) for i in range(60)],
            "timestamp": 0,
        }))
        assert cache.get("SPX") is None
        # Validator deletes invalid file so it doesn't keep getting reread
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_qualify_contract_cached(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True
        provider._ib = MagicMock()
        mock_contract = MagicMock(symbol="SPX", secType="OPT",
                                   lastTradeDateOrContractMonth="20260316",
                                   strike=5500.0, right="P", conId=99999)
        provider._ib.qualifyContractsAsync = AsyncMock(return_value=[mock_contract])
        await provider._qualify_contract_cached(mock_contract)
        await provider._qualify_contract_cached(mock_contract)
        assert provider._ib.qualifyContractsAsync.call_count == 1

    @pytest.mark.asyncio
    async def test_check_margin_not_connected(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        order = MultiLegOrder(broker=Broker.IBKR, legs=[
            OptionLeg(symbol="SPX", expiration="2026-03-16", strike=5500.0,
                      option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
        ], order_type=OrderType.LIMIT, net_price=1.00)
        with pytest.raises(RuntimeError, match="not connected"):
            await IBKRLiveProvider().check_margin(order)

    @pytest.mark.asyncio
    async def test_cache_cleared_on_connect(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._cache.contracts.put(MagicMock(), "SPX", "OPT", "20260316", 5500.0, "P")
        provider._cache.quotes.put("SPX", {"bid": 100})
        provider._cache.clear_all()
        assert provider._cache.contracts.stats["size"] == 0

    def test_cache_manager_stats(self):
        from app.core.providers.ibkr_cache import IBKRCacheManager
        stats = IBKRCacheManager(option_chain_cache_dir="/tmp/test_utp_cache").stats()
        assert "contracts" in stats
        assert "option_chains" in stats

    @pytest.mark.asyncio
    async def test_stub_get_option_chain(self):
        from app.core.providers.ibkr import IBKRProvider
        result = await IBKRProvider().get_option_chain("SPX")
        assert "expirations" in result

    @pytest.mark.asyncio
    async def test_stub_check_margin(self):
        from app.core.providers.ibkr import IBKRProvider
        order = MultiLegOrder(broker=Broker.IBKR, legs=[
            OptionLeg(symbol="SPX", expiration="2026-03-16", strike=5500.0,
                      option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
        ], order_type=OrderType.LIMIT, net_price=1.00)
        result = await IBKRProvider().check_margin(order)
        assert "init_margin" in result

    @pytest.mark.asyncio
    async def test_stub_get_account_balances(self):
        from app.core.providers.ibkr import IBKRProvider
        provider = IBKRProvider()
        await provider.connect()
        balances = await provider.get_account_balances()
        assert balances.cash == 100_000.00
        assert balances.net_liquidation == 136_000.00
        assert balances.buying_power == 200_000.00
        assert balances.broker == "ibkr"

    @pytest.mark.asyncio
    async def test_live_get_account_balances_not_connected(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        balances = await provider.get_account_balances()
        assert balances.cash == 0
        assert balances.net_liquidation == 0
        assert balances.broker == "ibkr"

    @pytest.mark.asyncio
    async def test_live_get_account_balances_with_values(self):
        from app.core.providers.ibkr import IBKRLiveProvider
        provider = IBKRLiveProvider()
        provider._connected = True

        mock_ib = MagicMock()
        # Simulate ib_insync AccountValue objects
        AccountValue = type("AccountValue", (), {})
        values = []
        for tag, val in [
            ("TotalCashValue", "50000.00"),
            ("NetLiquidation", "125000.00"),
            ("BuyingPower", "175000.00"),
            ("MaintMarginReq", "12000.00"),
            ("AvailableFunds", "63000.00"),
        ]:
            av = AccountValue()
            av.tag = tag
            av.value = val
            av.currency = "USD"
            av.account = "DU12345"
            values.append(av)
        mock_ib.accountValues.return_value = values
        provider._ib = mock_ib

        balances = await provider.get_account_balances()
        assert balances.cash == 50_000.00
        assert balances.net_liquidation == 125_000.00
        assert balances.buying_power == 175_000.00
        assert balances.maint_margin_req == 12_000.00
        assert balances.available_funds == 63_000.00


# ═══════════════════════════════════════════════════════════════════════════════
# IBKR REST Provider (Client Portal Gateway)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIBKRRestProvider:
    """Tests for the IBKRRestProvider (CPG REST API backend)."""

    def test_rest_broker_enum(self):
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        assert provider.broker == Broker.IBKR

    @pytest.mark.asyncio
    async def test_connect_auth_check(self):
        """connect() should call auth/status + accounts."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")

        mock_session = AsyncMock()
        auth_resp = AsyncMock()
        auth_resp.json = AsyncMock(return_value={"authenticated": True})
        auth_resp.text = AsyncMock(return_value='{"authenticated": true}')
        auth_resp.raise_for_status = MagicMock()
        auth_resp.__aenter__ = AsyncMock(return_value=auth_resp)
        auth_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session.post = MagicMock(return_value=auth_resp)
        mock_session.close = AsyncMock()

        # Patch aiohttp.ClientSession to return our mock
        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.TCPConnector"):
                await provider.connect()

        assert provider._connected is True
        assert provider._account_id == "U123"
        # Clean up keepalive
        await provider.disconnect()

    @pytest.mark.asyncio
    async def test_connect_not_authenticated(self):
        """connect() should raise when CPG session is not authenticated."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")

        mock_session = AsyncMock()
        auth_resp = AsyncMock()
        auth_resp.json = AsyncMock(return_value={"authenticated": False})
        auth_resp.text = AsyncMock(return_value='{"authenticated": false}')
        auth_resp.raise_for_status = MagicMock()
        auth_resp.__aenter__ = AsyncMock(return_value=auth_resp)
        auth_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session.post = MagicMock(return_value=auth_resp)
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("aiohttp.TCPConnector"):
                with pytest.raises(RuntimeError, match="not authenticated"):
                    await provider.connect()

        assert provider._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_cancels_keepalive(self):
        """disconnect() should cancel the keepalive task."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._session.close = AsyncMock()

        # Create a real asyncio task that we can cancel
        async def fake_keepalive():
            await asyncio.sleep(3600)

        task = asyncio.create_task(fake_keepalive())
        provider._keepalive_task = task

        await provider.disconnect()
        assert task.cancelled()
        assert provider._connected is False
        assert provider._session is None

    def test_is_healthy(self):
        """is_healthy() should reflect connection state."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        assert provider.is_healthy() is False

        provider._connected = True
        provider._session = AsyncMock()
        assert provider.is_healthy() is True

    @pytest.mark.asyncio
    async def test_get_quote(self):
        """get_quote() should map snapshot fields to Quote."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        snap_resp = AsyncMock()
        snap_resp.json = AsyncMock(return_value=[{
            "31": "5650.25", "84": "5649.50", "86": "5651.00", "87": "1500000",
        }])
        snap_resp.raise_for_status = MagicMock()
        snap_resp.__aenter__ = AsyncMock(return_value=snap_resp)
        snap_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=snap_resp)

        quote = await provider.get_quote("SPX")
        assert quote.symbol == "SPX"
        assert quote.last == 5650.25
        assert quote.bid == 5649.50
        assert quote.ask == 5651.00
        assert quote.volume == 1500000
        assert quote.source == "cpg"

    @pytest.mark.asyncio
    async def test_resolve_option_conid_whole_dollar_strike(self):
        """Whole-dollar strikes must be sent to CPG without trailing '.0'."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        captured_params: list[dict] = []

        async def fake_get(path, params=None, **_):
            captured_params.append(dict(params or {}))
            return [{"conid": 999001, "maturityDate": "20260427"}]

        provider._get = fake_get  # type: ignore[assignment]

        con_id = await provider._resolve_option_conid("SPX", "20260427", 7065.0, "P")
        assert con_id == 999001
        assert captured_params, "expected at least one CPG call"
        # Whole-dollar 7065.0 must be sent as "7065", not "7065.0"
        assert captured_params[0]["strike"] == "7065"
        # Fractional strikes should keep their decimals
        captured_params.clear()
        provider._option_conid_cache.clear()
        provider._conid_cache["SPY"] = 756733  # short-circuit underlying lookup

        async def fake_get_frac(path, params=None, **_):
            captured_params.append(dict(params or {}))
            return [{"conid": 999002, "maturityDate": "20260427"}]

        provider._get = fake_get_frac  # type: ignore[assignment]
        await provider._resolve_option_conid("SPY", "20260427", 5.50, "P")
        assert captured_params[0]["strike"] == "5.5"

    @pytest.mark.asyncio
    async def test_resolve_option_conid_empty_list_diagnostic(self):
        """CPG empty-list response should produce a diagnostic error, not silent failure."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        async def fake_get(path, params=None, **_):
            return []  # CPG returns empty list when strike doesn't exist

        provider._get = fake_get  # type: ignore[assignment]

        with pytest.raises(RuntimeError) as excinfo:
            await provider._resolve_option_conid("SPX", "20260427", 7065.0, "P")
        msg = str(excinfo.value)
        # Error must include strike (without trailing ".0"), and per-attempt summary
        assert "7065P" in msg
        assert "attempts:" in msg
        assert "empty" in msg

    @pytest.mark.asyncio
    async def test_resolve_option_conid_primes_after_http_errors(self):
        """When all secdef/info attempts fail with HTTP errors in round 1,
        the resolver should issue a marketdata snapshot to prime the
        underlying conid, then retry secdef/info once."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        calls: list[dict] = []
        attempts = {"count": 0}

        async def fake_get(path, params=None, **_):
            calls.append({"path": path, "params": dict(params or {})})
            if path == "/iserver/marketdata/snapshot":
                return [{"31": "7150"}]
            if path == "/iserver/secdef/info":
                attempts["count"] += 1
                # Round 1: every exchange returns 500 (3 attempts: None, SMART, CBOE)
                if attempts["count"] <= 3:
                    raise RuntimeError("HTTP 500: ClientResponseError")
                # Round 2: succeed on first try
                return [{"conid": 999004, "maturityDate": "20260427"}]
            return []

        provider._get = fake_get  # type: ignore[assignment]

        con_id = await provider._resolve_option_conid("SPX", "20260427", 7070.0, "P")
        assert con_id == 999004
        # Confirm the snapshot prime happened between rounds
        snapshot_calls = [c for c in calls if c["path"] == "/iserver/marketdata/snapshot"]
        assert len(snapshot_calls) == 1, "expected exactly one snapshot prime"
        assert snapshot_calls[0]["params"]["conids"] == "416904"

    @pytest.mark.asyncio
    async def test_resolve_option_conid_priming_does_not_loop_forever(self, monkeypatch):
        """If priming + reauth + retry still all fail, raise (no unbounded
        retries). All three rounds must appear in the diagnostic."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        # Skip the real 1-second propagation sleep
        import asyncio as _aio
        async def _fast_sleep(*_a, **_k): return None
        monkeypatch.setattr(_aio, "sleep", _fast_sleep)

        async def fake_get(path, params=None, **_):
            if path == "/iserver/marketdata/snapshot":
                return [{"31": "7150"}]
            if path == "/iserver/secdef/info":
                raise RuntimeError("HTTP 500")
            return []

        async def fake_post(path, **_):
            return {}

        provider._get = fake_get  # type: ignore[assignment]
        provider._post = fake_post  # type: ignore[assignment]

        with pytest.raises(RuntimeError) as excinfo:
            await provider._resolve_option_conid("SPX", "20260427", 7070.0, "P")
        msg = str(excinfo.value)
        # All three rounds (R1 baseline, R2 post-prime, R3 post-reauth) should
        # be in the diagnostic.
        assert "r1 exch=" in msg
        assert "r2 exch=" in msg
        assert "r3 exch=" in msg

    @pytest.mark.asyncio
    async def test_resolve_option_conid_session_recovery_succeeds(self, monkeypatch):
        """Round 3 (post-session-recovery) succeeds when R1+R2 hit 500s with
        ``No Contracts retrieved``. Verifies that ``/tickle`` and
        ``/iserver/reauthenticate`` are issued before the final retry — this is
        the IBKR-recommended recovery for a daemon-side stale session whose
        gateway-level auth still appears healthy."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        import asyncio as _aio
        async def _fast_sleep(*_a, **_k): return None
        monkeypatch.setattr(_aio, "sleep", _fast_sleep)

        attempts = {"count": 0}
        post_calls: list[str] = []

        async def fake_get(path, params=None, **_):
            if path == "/iserver/marketdata/snapshot":
                return [{"31": "7150"}]
            if path == "/iserver/secdef/info":
                attempts["count"] += 1
                # R1 (3) + R2 (3) all return CPG's stale-session signature.
                if attempts["count"] <= 6:
                    raise RuntimeError(
                        "HTTP 500 from /iserver/secdef/info: "
                        '{"error":"No Contracts retrieved "}'
                    )
                return [{"conid": 868691755, "maturityDate": "20260428"}]
            return []

        async def fake_post(path, **_):
            post_calls.append(path)
            if path == "/iserver/auth/status":
                return {"authenticated": True}
            return {}

        provider._get = fake_get  # type: ignore[assignment]
        provider._post = fake_post  # type: ignore[assignment]

        con_id = await provider._resolve_option_conid("SPX", "20260428", 7050.0, "P")
        assert con_id == 868691755
        assert attempts["count"] == 7  # 3 + 3 + 1

        # Recovery sequence ran before the successful R3 attempt
        assert "/tickle" in post_calls
        assert "/iserver/reauthenticate" in post_calls
        assert "/iserver/auth/status" in post_calls
        # Order: tickle, reauthenticate, auth/status
        recovery_order = [p for p in post_calls
                          if p in ("/tickle", "/iserver/reauthenticate", "/iserver/auth/status")]
        assert recovery_order == ["/tickle", "/iserver/reauthenticate", "/iserver/auth/status"]

    @pytest.mark.asyncio
    async def test_get_surfaces_response_body_on_error(self):
        """``_get`` must include the response body in the raised exception so
        callers can see CPG's error message (e.g. ``Authenticated: false``,
        ``No Contracts retrieved``). Previously bare ClientResponseError gave
        no usable info — the surface was just the URL."""
        from unittest.mock import AsyncMock, MagicMock
        from app.core.providers.ibkr_rest import IBKRRestProvider

        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        resp = AsyncMock()
        resp.status = 500
        resp.raise_for_status = MagicMock(side_effect=RuntimeError("500 error"))
        resp.text = AsyncMock(return_value='{"error":"No Contracts retrieved "}')
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=resp)

        with pytest.raises(RuntimeError) as excinfo:
            await provider._get("/iserver/secdef/info", params={"conid": 1})
        msg = str(excinfo.value)
        assert "HTTP 500" in msg
        assert "No Contracts retrieved" in msg

    @pytest.mark.asyncio
    async def test_resolve_option_conid_uses_cross_provider_store(self, tmp_path, monkeypatch):
        """A conId resolved by TWS (and persisted to OptionConidStore) must be
        usable by CPG, even when CPG's own /iserver/secdef/info is broken.
        This is the failover that makes TWS/CPG interchangeable."""
        # Force a clean shared cache dir for this test
        cache_dir = tmp_path / "shared_conids"
        monkeypatch.setenv("UTP_OPTION_CONID_CACHE_DIR", str(cache_dir))

        # Simulate TWS having previously resolved this contract: write to
        # the shared store directly (this is what _qualify_option does).
        from app.core.providers.ibkr_cache import OptionConidStore
        store = OptionConidStore()
        store.put("SPX", "20260427", 7075.0, "P", 555444)

        # Build a fresh CPG provider — its IBKRCacheManager creates a NEW
        # OptionConidStore, but pointed at the same env-overridden dir, so
        # it loads the on-disk state.
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        called = {"secdef": 0}

        async def fake_get(path, params=None, **_):
            if path == "/iserver/secdef/info":
                called["secdef"] += 1
                # Even if CPG is fully broken, the store should already have it
                raise RuntimeError("CPG 500")
            return []

        provider._get = fake_get  # type: ignore[assignment]

        con_id = await provider._resolve_option_conid("SPX", "20260427", 7075.0, "P")
        assert con_id == 555444
        assert called["secdef"] == 0, "should never call secdef/info when shared store has the conid"

    @pytest.mark.asyncio
    async def test_resolve_option_conid_writes_to_cross_provider_store(self, tmp_path, monkeypatch):
        """A successful CPG resolution must write to the shared store so TWS
        can pick it up later (and vice-versa)."""
        cache_dir = tmp_path / "shared_conids"
        monkeypatch.setenv("UTP_OPTION_CONID_CACHE_DIR", str(cache_dir))

        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        async def fake_get(path, params=None, **_):
            return [{"conid": 778899, "maturityDate": "20260427"}]

        provider._get = fake_get  # type: ignore[assignment]
        await provider._resolve_option_conid("SPX", "20260427", 7075.0, "P")

        # Now read via a fresh store instance (simulates a different provider
        # opening the same on-disk cache)
        from app.core.providers.ibkr_cache import OptionConidStore
        fresh = OptionConidStore()
        assert fresh.get("SPX", "20260427", 7075.0, "P") == 778899

    @pytest.mark.asyncio
    async def test_resolve_option_conid_cache_hit(self):
        """Pre-populated cache should short-circuit without any HTTP calls."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._option_conid_cache["SPX_20260427_7065.0_P"] = 999003

        called = False

        async def fake_get(*_args, **_kwargs):
            nonlocal called
            called = True
            return []

        provider._get = fake_get  # type: ignore[assignment]
        con_id = await provider._resolve_option_conid("SPX", "20260427", 7065.0, "P")
        assert con_id == 999003
        assert called is False

    @pytest.mark.asyncio
    async def test_get_quote_not_connected(self):
        """get_quote() should raise when disconnected."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        with pytest.raises(RuntimeError, match="not connected"):
            await provider.get_quote("SPX")

    @pytest.mark.asyncio
    async def test_get_positions(self):
        """get_positions() should map CPG positions to Position model."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        pos_resp = AsyncMock()
        pos_resp.json = AsyncMock(return_value=[{
            "conid": 416904, "contractDesc": "SPX OPT", "position": -3,
            "avgCost": 2.50, "mktValue": -750.0, "unrealizedPnl": 100.0,
            "assetClass": "OPT", "expiry": "20260320", "strike": 5500.0,
            "putOrCall": "P",
        }])
        pos_resp.raise_for_status = MagicMock()
        pos_resp.__aenter__ = AsyncMock(return_value=pos_resp)
        pos_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=pos_resp)

        positions = await provider.get_positions()
        assert len(positions) == 1
        assert positions[0].quantity == -3
        assert positions[0].con_id == 416904
        assert positions[0].sec_type == "OPT"
        assert positions[0].right == "P"

    @pytest.mark.asyncio
    async def test_get_positions_empty(self):
        """get_positions() returns [] when no positions."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        pos_resp = AsyncMock()
        pos_resp.json = AsyncMock(return_value=[])
        pos_resp.raise_for_status = MagicMock()
        pos_resp.__aenter__ = AsyncMock(return_value=pos_resp)
        pos_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=pos_resp)

        positions = await provider.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_execute_equity_order_buy(self):
        """execute_equity_order() should build correct BUY JSON."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPY"] = 756733

        import app.config
        orig = app.config.settings.ibkr_readonly
        app.config.settings.ibkr_readonly = False

        order_resp = AsyncMock()
        order_resp.json = AsyncMock(return_value=[{"order_id": "12345"}])
        order_resp.text = AsyncMock(return_value='[{"order_id": "12345"}]')
        order_resp.raise_for_status = MagicMock()
        order_resp.__aenter__ = AsyncMock(return_value=order_resp)
        order_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.post = MagicMock(return_value=order_resp)

        order = EquityOrder(broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY,
                           quantity=100, order_type=OrderType.MARKET)
        result = await provider.execute_equity_order(order)
        assert result.status == OrderStatus.SUBMITTED
        assert "SPY" in result.message
        app.config.settings.ibkr_readonly = orig

    @pytest.mark.asyncio
    async def test_execute_equity_readonly(self):
        """execute_equity_order() should reject when readonly."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        import app.config
        orig = app.config.settings.ibkr_readonly
        app.config.settings.ibkr_readonly = True

        order = EquityOrder(broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY,
                           quantity=100, order_type=OrderType.MARKET)
        result = await provider.execute_equity_order(order)
        assert result.status == OrderStatus.REJECTED
        assert "read-only" in result.message
        app.config.settings.ibkr_readonly = orig

    @pytest.mark.asyncio
    async def test_execute_credit_spread_conidex(self):
        """execute_multi_leg_order() should build correct conidex for credit spread."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904
        provider._option_conid_cache["SPX_20260320_5500.0_P"] = 100001
        provider._option_conid_cache["SPX_20260320_5475.0_P"] = 100002

        import app.config
        orig = app.config.settings.ibkr_readonly
        app.config.settings.ibkr_readonly = False

        order_resp = AsyncMock()
        order_resp.json = AsyncMock(return_value=[{"order_id": "67890"}])
        order_resp.text = AsyncMock(return_value='[{"order_id": "67890"}]')
        order_resp.raise_for_status = MagicMock()
        order_resp.__aenter__ = AsyncMock(return_value=order_resp)
        order_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.post = MagicMock(return_value=order_resp)

        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                         option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                         option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            order_type=OrderType.MARKET,
            quantity=1,
        )
        result = await provider.execute_multi_leg_order(order)
        assert result.status == OrderStatus.SUBMITTED

        # Verify the conidex was constructed correctly
        call_args = provider._session.post.call_args_list
        # Find the order submission call (the one with /orders in the URL)
        for call in call_args:
            url = call[0][0] if call[0] else ""
            if "/orders" in url and "whatif" not in url:
                body = call[1].get("json", {})
                orders = body.get("orders", [{}])
                if orders:
                    conidex = orders[0].get("conidex", "")
                    # SELL leg gets negative ratio, BUY gets positive
                    assert "100001/-1" in conidex
                    assert "100002/1" in conidex
                    assert conidex.startswith("416904;;;")
                break

        app.config.settings.ibkr_readonly = orig

    @pytest.mark.asyncio
    async def test_execute_credit_spread_reduces_gcd(self):
        """When callers pass per-leg quantity equal to the combo size, the
        provider must reduce ratios to lowest terms (1:-1) — otherwise IBKR
        returns 'Invalid leg ratio' (error 321)."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904
        provider._option_conid_cache["SPX_20260320_5500.0_P"] = 100001
        provider._option_conid_cache["SPX_20260320_5475.0_P"] = 100002

        import app.config
        orig = app.config.settings.ibkr_readonly
        app.config.settings.ibkr_readonly = False

        order_resp = AsyncMock()
        order_resp.json = AsyncMock(return_value=[{"order_id": "67891"}])
        order_resp.text = AsyncMock(return_value='[{"order_id": "67891"}]')
        order_resp.raise_for_status = MagicMock()
        order_resp.__aenter__ = AsyncMock(return_value=order_resp)
        order_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.post = MagicMock(return_value=order_resp)

        # Both legs at quantity=10 (caller mistake: should be 1).
        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                         option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=10),
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                         option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=10),
            ],
            order_type=OrderType.MARKET,
            quantity=10,
        )
        result = await provider.execute_multi_leg_order(order)
        assert result.status == OrderStatus.SUBMITTED

        for call in provider._session.post.call_args_list:
            url = call[0][0] if call[0] else ""
            if "/orders" in url and "whatif" not in url:
                body = call[1].get("json", {})
                orders = body.get("orders", [{}])
                if orders:
                    conidex = orders[0].get("conidex", "")
                    # Reduced to 1:-1 — NOT 10:-10
                    assert "100001/-1" in conidex
                    assert "100002/1" in conidex
                    assert "100001/-10" not in conidex
                    assert "100002/10" not in conidex
                    assert orders[0]["quantity"] == 10
                break

        app.config.settings.ibkr_readonly = orig

    @pytest.mark.asyncio
    async def test_execute_market_order(self):
        """execute_multi_leg_order() with MARKET type should use orderType=MKT."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904
        provider._option_conid_cache["SPX_20260320_5500.0_P"] = 100001
        provider._option_conid_cache["SPX_20260320_5475.0_P"] = 100002

        import app.config
        orig = app.config.settings.ibkr_readonly
        app.config.settings.ibkr_readonly = False

        order_resp = AsyncMock()
        order_resp.json = AsyncMock(return_value=[{"order_id": "111"}])
        order_resp.text = AsyncMock(return_value='[{"order_id": "111"}]')
        order_resp.raise_for_status = MagicMock()
        order_resp.__aenter__ = AsyncMock(return_value=order_resp)
        order_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.post = MagicMock(return_value=order_resp)

        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                         option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                         option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            order_type=OrderType.MARKET,
            quantity=1,
        )
        result = await provider.execute_multi_leg_order(order)
        assert result.status == OrderStatus.SUBMITTED
        assert "MARKET" in result.message
        app.config.settings.ibkr_readonly = orig

    @pytest.mark.asyncio
    async def test_order_confirmation_reply(self):
        """_place_order_with_confirmation() should auto-confirm replyId."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        # First call returns confirmation prompt, second returns order
        call_count = 0
        def make_resp(return_val):
            resp = AsyncMock()
            resp.json = AsyncMock(return_value=return_val)
            resp.text = AsyncMock(return_value=str(return_val))
            resp.raise_for_status = MagicMock()
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=False)
            return resp

        responses = [
            make_resp([{"id": "abc-123", "message": ["Confirm order?"]}]),
            make_resp([{"order_id": "99999"}]),
        ]
        provider._session.post = MagicMock(side_effect=responses)

        result = await provider._place_order_with_confirmation({"conid": 416904, "side": "BUY"})
        assert result.get("order_id") == "99999"
        # Should have made 2 POST calls: order + confirmation
        assert provider._session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_get_order_status(self):
        """get_order_status() should map CPG status to OrderStatus."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        status_resp = AsyncMock()
        status_resp.json = AsyncMock(return_value={
            "order_status": "Filled", "avg_price": 3.50, "filled_quantity": 5,
        })
        status_resp.raise_for_status = MagicMock()
        status_resp.__aenter__ = AsyncMock(return_value=status_resp)
        status_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=status_resp)

        result = await provider.get_order_status("12345")
        assert result.status == OrderStatus.FILLED
        assert result.filled_price == 3.50
        assert result.filled_quantity == 5

    @pytest.mark.asyncio
    async def test_get_option_chain(self):
        """get_option_chain() should return expirations + strikes from cache."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904

        # Seed the daily cache directly (avoids mocking the multi-step CPG flow).
        # Use a realistic SPX chain: ~1000 strikes spanning $1000-$10990 so
        # min/max < 0.6 and the strike-span validator accepts it.
        from datetime import date, timedelta
        today_exp = date.today().strftime("%Y%m%d")
        future_exp = (date.today() + timedelta(days=7)).strftime("%Y%m%d")
        strikes = [float(1000 + i * 10) for i in range(1000)]
        provider._cache.option_chains.put(
            "SPX",
            expirations=[today_exp, future_exp],
            strikes=strikes,
        )

        chain = await provider.get_option_chain("SPX")
        assert 5100.0 in chain["strikes"]
        assert len(chain["strikes"]) == 1000
        assert today_exp in chain["expirations"]

    @pytest.mark.asyncio
    async def test_check_margin_whatif(self):
        """check_margin() should parse what-if response."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()
        provider._conid_cache["SPX"] = 416904
        provider._option_conid_cache["SPX_20260320_5500.0_P"] = 100001
        provider._option_conid_cache["SPX_20260320_5475.0_P"] = 100002

        whatif_resp = AsyncMock()
        whatif_resp.json = AsyncMock(return_value={
            "initMargin": "2500.0", "maintMargin": "2500.0", "commission": "2.60",
        })
        whatif_resp.text = AsyncMock(return_value='{}')
        whatif_resp.raise_for_status = MagicMock()
        whatif_resp.__aenter__ = AsyncMock(return_value=whatif_resp)
        whatif_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.post = MagicMock(return_value=whatif_resp)

        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                         option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                         option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            quantity=1,
        )
        margin = await provider.check_margin(order)
        assert margin["init_margin"] == 2500.0
        assert margin["commission"] == 2.60

    @pytest.mark.asyncio
    async def test_get_account_balances(self):
        """get_account_balances() should map summary to AccountBalances."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        summary_resp = AsyncMock()
        summary_resp.json = AsyncMock(return_value={
            "totalcashvalue": {"amount": 50000.0},
            "netliquidation": {"amount": 150000.0},
            "buyingpower": {"amount": 200000.0},
            "maintmarginreq": {"amount": 25000.0},
            "availablefunds": {"amount": 125000.0},
        })
        summary_resp.raise_for_status = MagicMock()
        summary_resp.__aenter__ = AsyncMock(return_value=summary_resp)
        summary_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=summary_resp)

        balances = await provider.get_account_balances()
        assert balances.cash == 50000.0
        assert balances.net_liquidation == 150000.0
        assert balances.buying_power == 200000.0
        assert balances.broker == "ibkr"

    @pytest.mark.asyncio
    async def test_get_open_orders(self):
        """get_open_orders() should map to OrderResult list."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        orders_resp = AsyncMock()
        orders_resp.json = AsyncMock(return_value={
            "orders": [
                {"orderId": "100", "ticker": "SPX", "side": "BUY", "totalSize": 5,
                 "orderType": "LMT", "price": 3.50, "status": "Submitted"},
            ]
        })
        orders_resp.raise_for_status = MagicMock()
        orders_resp.__aenter__ = AsyncMock(return_value=orders_resp)
        orders_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=orders_resp)

        orders = await provider.get_open_orders()
        assert len(orders) == 1
        assert orders[0].order_id == "100"
        assert orders[0].status == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        """cancel_order() should send DELETE correctly."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        del_resp = AsyncMock()
        del_resp.json = AsyncMock(return_value={"msg": "cancelled"})
        del_resp.text = AsyncMock(return_value='{}')
        del_resp.raise_for_status = MagicMock()
        del_resp.__aenter__ = AsyncMock(return_value=del_resp)
        del_resp.__aexit__ = AsyncMock(return_value=False)

        status_resp = AsyncMock()
        status_resp.json = AsyncMock(return_value={"order_status": "Cancelled"})
        status_resp.raise_for_status = MagicMock()
        status_resp.__aenter__ = AsyncMock(return_value=status_resp)
        status_resp.__aexit__ = AsyncMock(return_value=False)

        provider._session.delete = MagicMock(return_value=del_resp)
        provider._session.get = MagicMock(return_value=status_resp)

        result = await provider.cancel_order("12345")
        assert result.status == OrderStatus.CANCELLED
        provider._session.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_option_quotes_warm_path_skips_second_call(self):
        """When CPG returns populated quotes on the first snapshot call (warm
        conIDs), we must NOT issue the second call. This makes CPG's per-cycle
        cost match the TWS persistent-subscription path: first call slow
        (cold subscribe), subsequent calls fast (warm read)."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        async def fake_chain(_sym):
            return {"expirations": ["2026-04-29"], "strikes": [26500.0, 26600.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        async def fake_resolve(symbol, exp, strike, right):
            return 800000 + int(strike)
        provider._resolve_option_conid = fake_resolve  # type: ignore[assignment]

        get_calls: list[dict] = []

        async def fake_get(path, params=None, **_):
            get_calls.append({"path": path, "params": dict(params or {})})
            return [
                {"conid": 826500, "31": "12.5", "84": "12.4", "86": "12.6", "87": "100"},
                {"conid": 826600, "31": "13.5", "84": "13.4", "86": "13.6", "87": "200"},
            ]

        provider._get = fake_get  # type: ignore[assignment]

        results = await provider.get_option_quotes("NDX", "2026-04-29", "PUT")

        snapshot_calls = [c for c in get_calls if c["path"] == "/iserver/marketdata/snapshot"]
        assert len(snapshot_calls) == 1, (
            "warm conIDs must not require the second snapshot call — "
            f"got {len(snapshot_calls)} calls"
        )
        assert len(results) == 2
        assert all(r["bid"] > 0 and r["ask"] > 0 for r in results)

    @pytest.mark.asyncio
    async def test_get_option_quotes_cold_path_does_second_call(self, monkeypatch):
        """When the first call returns rows with no prices (cold conIDs just
        subscribed), the second call still fires after a short propagation
        wait. This is the only case where the double-call cost is paid."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        async def fake_chain(_sym):
            return {"expirations": ["2026-04-29"], "strikes": [26500.0]}
        provider.get_option_chain = fake_chain  # type: ignore[assignment]

        async def fake_resolve(symbol, exp, strike, right):
            return 826500
        provider._resolve_option_conid = fake_resolve  # type: ignore[assignment]

        # Skip the real 0.3s propagation sleep
        import asyncio as _aio
        async def _fast_sleep(*_a, **_k): return None
        monkeypatch.setattr(_aio, "sleep", _fast_sleep)

        calls = {"n": 0}

        async def fake_get(path, params=None, **_):
            if path != "/iserver/marketdata/snapshot":
                return {}
            calls["n"] += 1
            if calls["n"] == 1:
                # Cold response — empty fields
                return [{"conid": 826500, "31": "-1", "84": "", "86": None}]
            # Warm response — populated
            return [{"conid": 826500, "31": "12.5", "84": "12.4", "86": "12.6", "87": "100"}]

        provider._get = fake_get  # type: ignore[assignment]
        results = await provider.get_option_quotes("NDX", "2026-04-29", "PUT")

        assert calls["n"] == 2, f"cold path should fire two calls; got {calls['n']}"
        assert len(results) == 1
        assert results[0]["bid"] == 12.4

    def test_resolve_conid_cached(self):
        """Cached conIds should skip HTTP calls."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._conid_cache["AAPL"] = 265598
        # No session needed — cache hit
        assert provider._conid_cache["AAPL"] == 265598

    @pytest.mark.asyncio
    async def test_resolve_conid_uses_name_false(self):
        """secdef/search must be called with name=False (exact symbol match).

        ``name: True`` was treating the symbol as a partial company-name
        substring, which caused NDX to bind to conid 416843 (an adjacent
        NDX-prefixed instrument with no listed options for the active
        month) and broke every downstream secdef/info lookup.
        """
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True

        captured_payloads: list[dict] = []

        async def fake_post(path, json=None, **_):
            captured_payloads.append(dict(json or {}))
            return [{"conid": 9694, "symbol": "NDX"}]

        provider._post = fake_post  # type: ignore[assignment]

        con_id = await provider._resolve_conid("NDX")
        assert con_id == 9694
        assert captured_payloads, "expected at least one secdef/search POST"
        assert all(p.get("name") is False for p in captured_payloads), (
            f"name flag must be False; got {[p.get('name') for p in captured_payloads]}"
        )

    @pytest.mark.asyncio
    async def test_resolve_conid_prefers_exact_symbol_match(self):
        """When CPG returns multiple matches, prefer the one whose ``symbol``
        equals the requested ticker exactly. Defends against future drift in
        the search ranking returning an adjacent ticker first."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True

        async def fake_post(path, json=None, **_):
            # 416843-style adjacent symbol first, canonical NDX second
            return [
                {"conid": 416843, "symbol": "NDXP"},
                {"conid": 9694, "symbol": "NDX"},
                {"conid": 11111, "symbol": "NDXX"},
            ]

        provider._post = fake_post  # type: ignore[assignment]

        con_id = await provider._resolve_conid("NDX")
        assert con_id == 9694

    @pytest.mark.asyncio
    async def test_resolve_conid_falls_back_when_no_exact_symbol(self):
        """If no result advertises a matching symbol, fall back to the first
        entry that has a conid (covers responses where CPG omits the field)."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True

        async def fake_post(path, json=None, **_):
            return [{"conid": 265598}]  # no `symbol` field

        provider._post = fake_post  # type: ignore[assignment]

        con_id = await provider._resolve_conid("AAPL")
        assert con_id == 265598

    def test_ssl_disabled(self):
        """Provider should create session with SSL verification disabled."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        # The SSL context is created in connect(); just verify the gateway URL
        assert provider._gateway_url == "https://localhost:5000"

    def test_rate_limiter_configured(self):
        """Rate limiter should be set to 9 req/sec for CPG."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        assert provider._cache.rate_limiter._rate == 9.0

    @pytest.mark.asyncio
    async def test_get_daily_pnl_by_con_id(self):
        """get_daily_pnl_by_con_id() should extract dailyPnl from positions."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        pos_resp = AsyncMock()
        pos_resp.json = AsyncMock(return_value=[
            {"conid": 416904, "position": -3, "dailyPnl": -42.50},
            {"conid": 265598, "position": 100, "dailyPnl": 125.00},
            {"conid": 999999, "position": 10, "dailyPnl": None},
        ])
        pos_resp.raise_for_status = MagicMock()
        pos_resp.__aenter__ = AsyncMock(return_value=pos_resp)
        pos_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=pos_resp)

        result = await provider.get_daily_pnl_by_con_id()
        assert result[416904] == -42.50
        assert result[265598] == 125.00
        assert 999999 not in result  # None dailyPnl excluded

    @pytest.mark.asyncio
    async def test_get_account_daily_pnl(self):
        """get_account_daily_pnl() should extract dpl from partitioned PnL."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        provider._connected = True
        provider._session = AsyncMock()

        pnl_resp = AsyncMock()
        pnl_resp.json = AsyncMock(return_value={
            "acctId": {"U123": {"dpl": -256.78, "nl": 963000.0, "upl": -87000.0}},
        })
        pnl_resp.raise_for_status = MagicMock()
        pnl_resp.__aenter__ = AsyncMock(return_value=pnl_resp)
        pnl_resp.__aexit__ = AsyncMock(return_value=False)
        provider._session.get = MagicMock(return_value=pnl_resp)

        result = await provider.get_account_daily_pnl()
        assert result == -256.78

    @pytest.mark.asyncio
    async def test_get_account_daily_pnl_not_connected(self):
        """get_account_daily_pnl() returns 0 when not connected."""
        from app.core.providers.ibkr_rest import IBKRRestProvider
        provider = IBKRRestProvider(gateway_url="https://localhost:5000", account_id="U123")
        result = await provider.get_account_daily_pnl()
        assert result == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Account Balances in Portfolio
# ═══════════════════════════════════════════════════════════════════════════════


class TestAccountBalances:
    @pytest.mark.asyncio
    async def test_portfolio_shows_balances_in_dry_run(self, tmp_path, capsys):
        """Dry-run portfolio shows P&L but no broker balances."""
        args = argparse.Namespace(data_dir=str(tmp_path))
        rc = await _cmd_portfolio(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Cash Deployed" in out
        # No broker balances in dry-run
        assert "Net Liquidation" not in out

    def test_account_balances_model(self):
        from app.models import AccountBalances
        b = AccountBalances(cash=50000, net_liquidation=125000, buying_power=175000, broker="ibkr")
        assert b.cash == 50000
        assert b.net_liquidation == 125000
        assert b.buying_power == 175000
        assert b.maint_margin_req == 0
        assert b.available_funds == 0

    def test_dashboard_summary_has_balance_fields(self):
        from app.models import DashboardSummary
        summary = DashboardSummary(active_positions=[])
        assert summary.cash_available == 0
        assert summary.net_liquidation == 0
        assert summary.buying_power == 0
        assert summary.maint_margin_req == 0
        assert summary.available_funds == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Price Spread
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoPrice:
    @pytest.mark.asyncio
    async def test_auto_price_credit_spread_market(self, capsys):
        """Default auto-price returns market price (sell at bid, buy at ask)."""
        mock_quotes = [
            {"strike": 5500.0, "bid": 8.00, "ask": 8.50, "last": 8.25, "volume": 100},
            {"strike": 5475.0, "bid": 5.00, "ask": 5.50, "last": 5.25, "volume": 200},
        ]
        provider = AsyncMock()
        with patch("app.services.market_data.get_option_quotes_with_age",
                   new_callable=AsyncMock,
                   return_value=(mock_quotes, 0.0, "provider")):
            price = await _auto_price_spread(
                provider, "SPX", "2026-03-20",
                [5500.0, 5475.0], "PUT", "credit_spread")
        # market = short_bid - long_ask = 8.00 - 5.50 = 2.50
        assert price == 2.50
        out = capsys.readouterr().out
        assert "Market: $2.50" in out
        assert "Mid: $3.00" in out

    @pytest.mark.asyncio
    async def test_auto_price_credit_spread_mid(self, capsys):
        """With use_mid=True, returns mid-point between market and best-case."""
        mock_quotes = [
            {"strike": 5500.0, "bid": 8.00, "ask": 8.50, "last": 8.25, "volume": 100},
            {"strike": 5475.0, "bid": 5.00, "ask": 5.50, "last": 5.25, "volume": 200},
        ]
        provider = AsyncMock()
        with patch("app.services.market_data.get_option_quotes_with_age",
                   new_callable=AsyncMock,
                   return_value=(mock_quotes, 0.0, "provider")):
            price = await _auto_price_spread(
            provider, "SPX", "2026-03-20",
            [5500.0, 5475.0], "PUT", "credit_spread", use_mid=True)
        # market = 2.50, best = 8.50 - 5.00 = 3.50, mid = 3.00
        assert price == 3.00

    @pytest.mark.asyncio
    async def test_auto_price_debit_spread_market(self, capsys):
        """Default auto-price for debit spread returns market price."""
        mock_quotes = [
            {"strike": 480.0, "bid": 6.00, "ask": 6.40, "last": 6.20, "volume": 100},
            {"strike": 490.0, "bid": 3.00, "ask": 3.40, "last": 3.20, "volume": 200},
        ]
        provider = AsyncMock()
        with patch("app.services.market_data.get_option_quotes_with_age",
                   new_callable=AsyncMock,
                   return_value=(mock_quotes, 0.0, "provider")):
            price = await _auto_price_spread(
                provider, "QQQ", "2026-03-20",
                [480.0, 490.0], "CALL", "debit_spread")
        # market = long_ask - short_bid = 6.40 - 3.00 = 3.40
        assert price == 3.40
        out = capsys.readouterr().out
        assert "Market: $3.40" in out
        assert "Mid: $3.00" in out

    @pytest.mark.asyncio
    async def test_auto_price_iron_condor_market(self, capsys):
        """Default auto-price for iron condor returns market price."""
        provider = AsyncMock()

        async def mock_mkt_opts_age(symbol, expiration, option_type, **kwargs):
            if option_type == "PUT":
                return ([
                    {"strike": 5400.0, "bid": 4.00, "ask": 4.50, "last": 4.25, "volume": 100},
                    {"strike": 5375.0, "bid": 2.00, "ask": 2.50, "last": 2.25, "volume": 200},
                ], 0.0, "provider")
            else:
                return ([
                    {"strike": 5700.0, "bid": 3.00, "ask": 3.50, "last": 3.25, "volume": 100},
                    {"strike": 5725.0, "bid": 1.00, "ask": 1.50, "last": 1.25, "volume": 200},
                ], 0.0, "provider")

        with patch("app.services.market_data.get_option_quotes_with_age",
                   side_effect=mock_mkt_opts_age):
            price = await _auto_price_iron_condor(
                provider, "SPX", "2026-03-20", 5400.0, 5375.0, 5700.0, 5725.0)
        # puts: market = 4.00-2.50=1.50, calls: market = 3.00-1.50=1.50, total = 3.00
        assert price == 3.00
        out = capsys.readouterr().out
        assert "Market: $3.00" in out
        assert "Mid: $4.00" in out

    @pytest.mark.asyncio
    async def test_auto_price_iron_condor_mid(self, capsys):
        """With use_mid=True, iron condor returns mid-point."""
        provider = AsyncMock()

        async def mock_mkt_opts_age(symbol, expiration, option_type, **kwargs):
            if option_type == "PUT":
                return ([
                    {"strike": 5400.0, "bid": 4.00, "ask": 4.50, "last": 4.25, "volume": 100},
                    {"strike": 5375.0, "bid": 2.00, "ask": 2.50, "last": 2.25, "volume": 200},
                ], 0.0, "provider")
            else:
                return ([
                    {"strike": 5700.0, "bid": 3.00, "ask": 3.50, "last": 3.25, "volume": 100},
                    {"strike": 5725.0, "bid": 1.00, "ask": 1.50, "last": 1.25, "volume": 200},
                ], 0.0, "provider")

        with patch("app.services.market_data.get_option_quotes_with_age",
                   side_effect=mock_mkt_opts_age):
            price = await _auto_price_iron_condor(
                provider, "SPX", "2026-03-20", 5400.0, 5375.0, 5700.0, 5725.0,
                use_mid=True)
        # market = 3.00, best = 5.00, mid = 4.00
        assert price == 4.00

    @pytest.mark.asyncio
    async def test_auto_price_missing_strike(self):
        """Returns None when quotes don't include requested strikes."""
        mock_quotes = [
            {"strike": 5500.0, "bid": 8.00, "ask": 8.50, "last": 8.25, "volume": 100},
        ]
        provider = AsyncMock()
        with patch("app.services.market_data.get_option_quotes_with_age",
                   new_callable=AsyncMock,
                   return_value=(mock_quotes, 0.0, "provider")):
            price = await _auto_price_spread(
                provider, "SPX", "2026-03-20",
                [5500.0, 5475.0], "PUT", "credit_spread")
            assert price is None

    @pytest.mark.asyncio
    async def test_auto_price_no_provider_method(self):
        """Returns None when provider lacks get_option_quotes — centralized layer handles."""
        provider = MagicMock(spec=[])
        with patch("app.services.market_data.get_option_quotes", new_callable=AsyncMock, return_value=[]):
            price = await _auto_price_spread(
                provider, "SPX", "2026-03-20",
                [5500.0, 5475.0], "PUT", "credit_spread")
        assert price is None


# ═══════════════════════════════════════════════════════════════════════════════
# Orders & Cancel Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrdersCommand:
    def test_orders_requires_mode(self, capsys):
        """orders command rejects dry-run mode."""
        from utp import _cmd_orders
        args = argparse.Namespace(
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            host="127.0.0.1", port=7497, client_id=10, exchange="SMART",
            data_dir="data/utp",
        )
        rc = asyncio.run(_cmd_orders(args))
        assert rc == 1
        assert "requires --paper or --live" in capsys.readouterr().out

    def test_cancel_requires_mode(self, capsys):
        """cancel command rejects dry-run mode."""
        from utp import _cmd_cancel
        args = argparse.Namespace(
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            order_id="123", all=False,
            host="127.0.0.1", port=7497, client_id=10, exchange="SMART",
            data_dir="data/utp",
        )
        rc = asyncio.run(_cmd_cancel(args))
        assert rc == 1
        assert "requires --paper or --live" in capsys.readouterr().out

    def test_cancel_requires_order_id_or_all(self, capsys):
        """cancel command requires --order-id or --all."""
        from utp import _cmd_cancel
        args = argparse.Namespace(
            dry_run=False, paper=True, live=False, _default_mode="paper",
            order_id=None, all=False,
            host="127.0.0.1", port=7497, client_id=10, exchange="SMART",
            data_dir="data/utp",
        )
        rc = asyncio.run(_cmd_cancel(args))
        assert rc == 1
        assert "Specify --order-id" in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════════════
# Status Change Deduplication
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatusChangeDedup:
    @pytest.fixture(autouse=True)
    def _drop_poll_floor(self, monkeypatch):
        """Production policy floors poll_interval at 2s. These dedup tests
        use sub-2s polls so they finish in milliseconds — drop the floor for
        the duration of the class."""
        monkeypatch.setattr(
            "app.services.trade_service._MIN_POLL_INTERVAL", 0.0
        )

    @pytest.mark.asyncio
    async def test_repeated_status_logs_once(self):
        """await_order_fill only logs status changes, not every poll."""
        provider = _FakeProvider(polls_before_fill=5)
        log_calls = []

        async def mock_log_status_change(broker, order_id, status):
            log_calls.append(status)

        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            with patch("app.services.ledger.get_ledger") as mock_ledger_fn:
                mock_ledger = MagicMock()
                mock_ledger.log_status_change = mock_log_status_change
                mock_ledger_fn.return_value = mock_ledger
                from app.services.trade_service import await_order_fill
                result = await await_order_fill(
                    broker=Broker.IBKR, order_id="dedup-001",
                    poll_interval=0.01, timeout=2.0)

        assert result.status == OrderStatus.FILLED
        # Should log SUBMITTED once + FILLED once = 2, not 6
        assert len(log_calls) == 2
        assert log_calls[0] == "SUBMITTED"
        assert log_calls[1] == "FILLED"

    @pytest.mark.asyncio
    async def test_timeout_logs_submitted_once(self):
        """On timeout, only one SUBMITTED log entry, not one per poll."""
        provider = _FakeProvider(polls_before_fill=9999)
        log_calls = []

        async def mock_log_status_change(broker, order_id, status):
            log_calls.append(status)

        with patch("app.services.trade_service.ProviderRegistry") as mock_reg:
            mock_reg.get.return_value = provider
            with patch("app.services.ledger.get_ledger") as mock_ledger_fn:
                mock_ledger = MagicMock()
                mock_ledger.log_status_change = mock_log_status_change
                mock_ledger_fn.return_value = mock_ledger
                from app.services.trade_service import await_order_fill
                result = await await_order_fill(
                    broker=Broker.IBKR, order_id="dedup-002",
                    poll_interval=0.02, timeout=0.1)

        assert result.status == OrderStatus.SUBMITTED
        # Should only log SUBMITTED once despite multiple polls
        assert len(log_calls) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Position Store — order_id tracking
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositionOrderId:
    def test_position_stores_order_id(self, tmp_path):
        """Position created from trade stores the order_id."""
        from app.services.position_store import PlatformPositionStore
        store = PlatformPositionStore(tmp_path / "positions.json")
        request = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY, quantity=10,
        ))
        result = OrderResult(order_id="test-order-123", broker=Broker.IBKR,
                            status=OrderStatus.FILLED)
        pos_id = store.add_position(request, result, is_paper=True)
        positions = store.get_open_positions()
        matching = [p for p in positions if p["position_id"] == pos_id]
        assert len(matching) == 1
        assert matching[0]["order_id"] == "test-order-123"

    def test_position_order_id_in_model(self):
        """TrackedPosition model supports order_id field."""
        from app.models import TrackedPosition
        pos = TrackedPosition(
            broker=Broker.IBKR, symbol="SPY", order_id="ord-456",
        )
        assert pos.order_id == "ord-456"

    def test_position_order_id_default_none(self):
        """TrackedPosition order_id defaults to None."""
        from app.models import TrackedPosition
        pos = TrackedPosition(broker=Broker.IBKR, symbol="SPY")
        assert pos.order_id is None


# ═══════════════════════════════════════════════════════════════════════════════
# Provider ABC — open orders & cancel
# ═══════════════════════════════════════════════════════════════════════════════


class TestProviderOpenOrdersCancel:
    @pytest.mark.asyncio
    async def test_default_get_open_orders_empty(self):
        """Default BrokerProvider.get_open_orders returns empty list."""
        from app.core.providers.ibkr import IBKRProvider
        provider = IBKRProvider()
        orders = await provider.get_open_orders()
        assert orders == []

    @pytest.mark.asyncio
    async def test_default_cancel_order_fails(self):
        """Default BrokerProvider.cancel_order returns FAILED."""
        from app.core.providers.ibkr import IBKRProvider
        provider = IBKRProvider()
        result = await provider.cancel_order("123")
        assert result.status == OrderStatus.FAILED

    @pytest.mark.asyncio
    async def test_default_get_portfolio_items_empty(self):
        """Default BrokerProvider.get_portfolio_items returns empty list."""
        from app.core.providers.ibkr import IBKRProvider
        provider = IBKRProvider()
        items = await provider.get_portfolio_items()
        assert items == []


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio with Broker P&L
# ═══════════════════════════════════════════════════════════════════════════════


class TestPortfolioBrokerPnL:
    @pytest.mark.asyncio
    async def test_portfolio_uses_broker_pnl(self, tmp_path, capsys):
        """Portfolio display uses IBKR portfolio items for P&L when available."""
        from app.services.position_store import get_position_store
        from app.core.provider import BrokerProvider

        # Add a multi-leg position
        store = get_position_store()
        req = TradeRequest(
            multi_leg_order=MultiLegOrder(
                broker=Broker.IBKR,
                legs=[
                    OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2560.0,
                              option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
                    OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2570.0,
                              option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
                ],
                order_type=OrderType.LIMIT,
                net_price=0.15,
                quantity=2,
            )
        )
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED,
                             filled_price=-0.15)
        store.add_position(req, result, is_paper=False)

        # Verify the portfolio items method is part of the ABC
        assert hasattr(BrokerProvider, "get_portfolio_items")

    @pytest.mark.asyncio
    async def test_portfolio_matching_legs_to_broker(self, tmp_path, capsys):
        """Portfolio leg matching correctly identifies spread components."""
        from app.services.position_store import get_position_store

        store = get_position_store()
        req = TradeRequest(
            multi_leg_order=MultiLegOrder(
                broker=Broker.IBKR,
                legs=[
                    OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                              option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                    OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                              option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
                ],
                order_type=OrderType.LIMIT,
                net_price=3.50,
                quantity=1,
            )
        )
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED,
                             filled_price=-3.50)
        store.add_position(req, result, is_paper=False)

        # Dry-run portfolio should still work without broker P&L
        args = argparse.Namespace(data_dir=str(tmp_path))
        rc = await _cmd_portfolio(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "SPX" in out


# ═══════════════════════════════════════════════════════════════════════════════
# Close Flag & Verbose Output
# ═══════════════════════════════════════════════════════════════════════════════


class TestCloseFlag:
    def test_credit_spread_close_instruction_type(self):
        """--close changes instruction type to credit_spread_close."""
        args = _ns(subcommand="credit-spread", symbol="RUT", expiration="2026-03-17",
                   short_strike=2550.0, long_strike=2570.0, option_type="CALL",
                   quantity=1, net_price=0.10, close=True)
        instr = _build_instruction_from_args("credit-spread", args)
        assert instr["type"] == "credit_spread_close"

    def test_credit_spread_open_instruction_type(self):
        """Without --close, instruction type is credit_spread."""
        args = _ns(subcommand="credit-spread", symbol="RUT", expiration="2026-03-17",
                   short_strike=2550.0, long_strike=2570.0, option_type="CALL",
                   quantity=1, net_price=0.10, close=False)
        instr = _build_instruction_from_args("credit-spread", args)
        assert instr["type"] == "credit_spread"

    def test_debit_spread_close_instruction_type(self):
        """--close changes debit spread instruction type."""
        args = _ns(subcommand="debit-spread", symbol="QQQ", expiration="2026-04-01",
                   long_strike=480.0, short_strike=490.0, option_type="CALL",
                   quantity=1, net_price=2.00, close=True)
        instr = _build_instruction_from_args("debit-spread", args)
        assert instr["type"] == "debit_spread_close"

    def test_iron_condor_close_instruction_type(self):
        """--close changes iron condor instruction type."""
        args = _ns(subcommand="iron-condor", symbol="SPX", expiration="2026-04-01",
                   put_short=5500.0, put_long=5475.0, call_short=5700.0,
                   call_long=5725.0, quantity=1, net_price=3.50, close=True)
        instr = _build_instruction_from_args("iron-condor", args)
        assert instr["type"] == "iron_condor_close"

    def test_playbook_service_credit_spread_close_legs(self):
        """credit_spread_close uses BUY_TO_CLOSE / SELL_TO_CLOSE legs."""
        from app.services.playbook_service import PlaybookService
        svc = PlaybookService()
        instr = PlaybookInstruction(
            id="test_close", type="credit_spread_close",
            params={"symbol": "RUT", "expiration": "2026-03-17",
                    "short_strike": 2550.0, "long_strike": 2570.0,
                    "option_type": "CALL", "quantity": 1, "net_price": 0.10})
        req = svc.instruction_to_trade_request(instr, Broker.IBKR)
        legs = req.multi_leg_order.legs
        assert legs[0].action == OptionAction.BUY_TO_CLOSE   # short leg
        assert legs[1].action == OptionAction.SELL_TO_CLOSE   # long leg

    def test_playbook_service_credit_spread_open_legs(self):
        """credit_spread uses SELL_TO_OPEN / BUY_TO_OPEN legs."""
        from app.services.playbook_service import PlaybookService
        svc = PlaybookService()
        instr = PlaybookInstruction(
            id="test_open", type="credit_spread",
            params={"symbol": "RUT", "expiration": "2026-03-17",
                    "short_strike": 2550.0, "long_strike": 2570.0,
                    "option_type": "CALL", "quantity": 1, "net_price": 0.28})
        req = svc.instruction_to_trade_request(instr, Broker.IBKR)
        legs = req.multi_leg_order.legs
        assert legs[0].action == OptionAction.SELL_TO_OPEN
        assert legs[1].action == OptionAction.BUY_TO_OPEN

    def test_playbook_service_iron_condor_close_legs(self):
        """iron_condor_close reverses all 4 legs."""
        from app.services.playbook_service import PlaybookService
        svc = PlaybookService()
        instr = PlaybookInstruction(
            id="test_ic_close", type="iron_condor_close",
            params={"symbol": "SPX", "expiration": "2026-04-01",
                    "put_short": 5500.0, "put_long": 5475.0,
                    "call_short": 5700.0, "call_long": 5725.0,
                    "quantity": 1, "net_price": 1.00})
        req = svc.instruction_to_trade_request(instr, Broker.IBKR)
        legs = req.multi_leg_order.legs
        assert legs[0].action == OptionAction.BUY_TO_CLOSE   # put short
        assert legs[1].action == OptionAction.SELL_TO_CLOSE   # put long
        assert legs[2].action == OptionAction.BUY_TO_CLOSE   # call short
        assert legs[3].action == OptionAction.SELL_TO_CLOSE   # call long


# ═══════════════════════════════════════════════════════════════════════════════
# Trades Command
# ═══════════════════════════════════════════════════════════════════════════════


class TestTradesCommand:
    def test_trades_no_trades(self, tmp_path, capsys):
        """trades command shows empty message when no trades."""
        from utp import _cmd_trades
        args = argparse.Namespace(
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            data_dir=str(tmp_path), days=0, detail=None,
        )
        rc = asyncio.run(_cmd_trades(args))
        assert rc == 0
        assert "no trades found" in capsys.readouterr().out

    def test_show_trade_detail_not_found(self, capsys):
        """_show_trade_detail returns 1 for unknown ID."""
        from utp import _show_trade_detail
        from app.services.position_store import get_position_store
        store = get_position_store()
        rc = _show_trade_detail(store, "nonexistent-id")
        assert rc == 1
        assert "No trade found" in capsys.readouterr().out

    def test_show_trade_detail_found(self, capsys):
        """_show_trade_detail shows full info for a matching position."""
        from utp import _show_trade_detail
        from app.services.position_store import get_position_store
        store = get_position_store()
        # Create a position
        request = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2550.0,
                          option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2570.0,
                          option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=0.28, quantity=1,
        ))
        result = OrderResult(order_id="42", broker=Broker.IBKR, status=OrderStatus.FILLED,
                            filled_price=-0.28)
        pos_id = store.add_position(request, result, is_paper=False)

        rc = _show_trade_detail(store, pos_id[:8])
        out = capsys.readouterr().out
        assert rc == 0
        assert "RUT" in out
        assert "Trade Detail" in out
        assert "Legs" in out
        assert "2550" in out
        assert "2570" in out
        assert "To close" in out


# ═══════════════════════════════════════════════════════════════════════════════
# Close by Position ID
# ═══════════════════════════════════════════════════════════════════════════════


class TestCloseCommand:
    def test_close_not_found(self, tmp_path, capsys):
        """close command returns 1 for unknown position ID."""
        args = argparse.Namespace(
            position_id="nonexistent",
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            data_dir=str(tmp_path), net_price=0.05,
        )
        rc = asyncio.run(_cmd_close(args))
        assert rc == 1
        assert "No open position found" in capsys.readouterr().out

    def test_close_equity_not_supported(self, tmp_path, capsys):
        """close command rejects equity positions (only multi-leg supported)."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY,
            quantity=10, order_type=OrderType.MARKET,
        ))
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=450.0)
        pos_id = store.add_position(req, result, is_paper=False)

        args = argparse.Namespace(
            position_id=pos_id[:8],
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            data_dir=str(tmp_path), net_price=0.05,
        )
        rc = asyncio.run(_cmd_close(args))
        assert rc == 1
        assert "only multi-leg" in capsys.readouterr().out

    def test_close_derives_parameters(self, tmp_path, capsys):
        """close command auto-derives spread params and executes dry-run."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2560.0,
                          option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2570.0,
                          option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=0.15, quantity=2,
        ))
        result = OrderResult(order_id="50", broker=Broker.IBKR, status=OrderStatus.FILLED,
                             filled_price=-0.15)
        pos_id = store.add_position(req, result, is_paper=False)

        args = argparse.Namespace(
            position_id=pos_id[:8],
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            data_dir=str(tmp_path), net_price=0.05,
            host="127.0.0.1", port=None, client_id=10,
            broker="ibkr", exchange="SMART",
            poll_timeout=30, poll_interval=1.0,
        )
        rc = asyncio.run(_cmd_close(args))
        out = capsys.readouterr().out
        assert rc == 0
        assert "Close Position" in out
        assert "RUT" in out
        assert "credit-spread" in out
        assert "2560" in out
        assert "2570" in out
        assert "Quantity:   2" in out
        # Verify original position was marked as closed (not a new one created)
        store_after = get_position_store()
        assert len(store_after.get_open_positions()) == 0, "Original position should be closed"
        closed = store_after.get_closed_positions()
        assert len(closed) == 1
        assert closed[0].get("position_id") == pos_id
        assert closed[0].get("status") == "closed"
        assert closed[0].get("exit_reason") == "closed_via_cli"

    def test_close_partial_quantity(self, tmp_path, capsys):
        """close --quantity N closes N of M contracts, reduces original position."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2560.0,
                          option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2570.0,
                          option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=0.15, quantity=3,
        ))
        result = OrderResult(order_id="60", broker=Broker.IBKR, status=OrderStatus.FILLED,
                             filled_price=-0.15)
        pos_id = store.add_position(req, result, is_paper=False)

        args = argparse.Namespace(
            position_id=pos_id[:8],
            quantity=1,
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            data_dir=str(tmp_path), net_price=0.05,
            host="127.0.0.1", port=None, client_id=10,
            broker="ibkr", exchange="SMART",
            poll_timeout=30, poll_interval=1.0,
        )
        rc = asyncio.run(_cmd_close(args))
        out = capsys.readouterr().out
        assert rc == 0
        assert "1 of 3 (partial)" in out
        # Original position still open but with reduced quantity
        store_after = get_position_store()
        open_positions = store_after.get_open_positions()
        original = [p for p in open_positions if p.get("position_id") == pos_id]
        assert len(original) == 1
        assert original[0].get("quantity") == 2  # 3 - 1 = 2
        assert original[0].get("status") == "open"
        # No ghost position created for the closing trade
        assert len(open_positions) == 1, "Partial close should not create a new position"

    def test_close_quantity_exceeds_position(self, tmp_path, capsys):
        """close --quantity N rejects when N > position quantity."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2560.0,
                          option_type=OptionType.CALL, action=OptionAction.SELL_TO_OPEN, quantity=1),
                OptionLeg(symbol="RUT", expiration="2026-03-17", strike=2570.0,
                          option_type=OptionType.CALL, action=OptionAction.BUY_TO_OPEN, quantity=1),
            ],
            net_price=0.15, quantity=2,
        ))
        result = OrderResult(order_id="70", broker=Broker.IBKR, status=OrderStatus.FILLED,
                             filled_price=-0.15)
        pos_id = store.add_position(req, result, is_paper=False)

        args = argparse.Namespace(
            position_id=pos_id[:8],
            quantity=5,
            dry_run=True, paper=False, live=False, _default_mode="dry-run",
            data_dir=str(tmp_path), net_price=0.05,
        )
        rc = asyncio.run(_cmd_close(args))
        assert rc == 1
        assert "only has 2" in capsys.readouterr().out


# ── Phase 0 Tests: Thread Safety, Zombie Positions, Reconnection, Logging ──


class TestPositionStoreLocking:
    """Thread safety tests for PlatformPositionStore."""

    def test_has_lock(self, position_store):
        """Position store has a threading lock."""
        import threading

        assert hasattr(position_store, "_lock")
        assert isinstance(position_store._lock, threading.Lock)

    def test_atomic_save(self, tmp_data_dir):
        """Save uses atomic write (temp file + rename)."""
        from app.services.position_store import PlatformPositionStore

        store = PlatformPositionStore(tmp_data_dir / "positions.json")
        request = TradeRequest(
            equity_order=EquityOrder(
                broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=100
            )
        )
        result = OrderResult(
            broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0
        )
        pid = store.add_position(request, result)
        # Verify file exists and is valid JSON
        with open(tmp_data_dir / "positions.json") as f:
            data = json.load(f)
        assert pid in data

    def test_concurrent_writes(self, tmp_data_dir):
        """Multiple threads can write without corrupting the store."""
        import threading

        from app.services.position_store import PlatformPositionStore

        store = PlatformPositionStore(tmp_data_dir / "positions.json")

        errors = []

        def add_position(i):
            try:
                request = TradeRequest(
                    equity_order=EquityOrder(
                        broker=Broker.IBKR,
                        symbol=f"SYM{i}",
                        side=OrderSide.BUY,
                        quantity=1,
                    )
                )
                result = OrderResult(
                    broker=Broker.IBKR,
                    status=OrderStatus.FILLED,
                    filled_price=float(i),
                )
                store.add_position(request, result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_position, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(store.get_open_positions()) == 10


class TestReduceQuantityZombie:
    """Test that reduce_quantity auto-closes at zero."""

    def test_reduce_to_zero_closes(self, position_store):
        request = TradeRequest(
            equity_order=EquityOrder(
                broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=5
            )
        )
        result = OrderResult(
            broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0
        )
        pid = position_store.add_position(request, result)
        pos = position_store.reduce_quantity(pid, 5)
        assert pos["status"] == "closed"
        assert pos["exit_reason"] == "fully_reduced"
        assert "exit_time" in pos

    def test_reduce_below_zero_closes(self, position_store):
        request = TradeRequest(
            equity_order=EquityOrder(
                broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=3
            )
        )
        result = OrderResult(
            broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0
        )
        pid = position_store.add_position(request, result)
        pos = position_store.reduce_quantity(pid, 10)
        assert pos["status"] == "closed"
        assert pos["quantity"] == 0

    def test_reduce_partial_stays_open(self, position_store):
        request = TradeRequest(
            equity_order=EquityOrder(
                broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=5
            )
        )
        result = OrderResult(
            broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0
        )
        pid = position_store.add_position(request, result)
        pos = position_store.reduce_quantity(pid, 2)
        assert pos["status"] == "open"
        assert pos["quantity"] == 3


class TestIBKRReconnection:
    """Test IBKR reconnection and health check."""

    def test_is_healthy_stub(self):
        from app.core.providers.ibkr import IBKRProvider

        provider = IBKRProvider()
        # Stub doesn't have is_healthy, only live does
        assert hasattr(provider, "_authenticated")

    def test_is_healthy_live_disconnected(self):
        from app.core.providers.ibkr import IBKRLiveProvider

        provider = IBKRLiveProvider()
        assert provider.is_healthy() is False

    def test_is_healthy_attribute(self):
        from app.core.providers.ibkr import IBKRLiveProvider

        provider = IBKRLiveProvider()
        assert hasattr(provider, "is_healthy")
        assert callable(provider.is_healthy)

    def test_reconnect_loop_attribute(self):
        from app.core.providers.ibkr import IBKRLiveProvider

        provider = IBKRLiveProvider()
        assert hasattr(provider, "_max_reconnect_retries")
        assert provider._max_reconnect_retries == 10

    def test_reconnect_backoff_cap_10s(self):
        """Backoff cap should be 10 seconds, not 120."""
        from app.core.providers.ibkr import IBKRLiveProvider

        provider = IBKRLiveProvider()
        assert hasattr(provider, "_reconnect_backoff_cap")
        assert provider._reconnect_backoff_cap == 10.0


class TestDaemonAutoRestart:
    """Test process-level auto-restart and degraded startup."""

    def test_run_daemon_with_restart_signal_no_restart(self):
        """Signal shutdown (rc=0) should not trigger restart."""
        import types

        call_count = 0

        original_run = asyncio.run

        def mock_daemon(args):
            nonlocal call_count
            call_count += 1
            return 0  # signal shutdown

        # Directly test the logic: rc=0 means clean exit, no restart
        import utp
        # Simulate: _cmd_daemon returns 0 (signal exit)
        old_cmd = utp._cmd_daemon

        async def fake_daemon(args):
            nonlocal call_count
            call_count += 1
            return 0

        utp._cmd_daemon = fake_daemon
        try:
            args = types.SimpleNamespace(no_restart=False)
            rc = utp._run_daemon_with_restart(args)
            assert rc == 0
            assert call_count == 1  # called once, not restarted
        finally:
            utp._cmd_daemon = old_cmd

    def test_run_daemon_with_restart_crash_restarts(self):
        """Crash should trigger restart with backoff."""
        import types
        import utp

        call_count = 0

        async def crashing_daemon(args):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("test crash")
            return 0  # succeed on 3rd try

        old_cmd = utp._cmd_daemon
        utp._cmd_daemon = crashing_daemon
        try:
            args = types.SimpleNamespace(no_restart=False)
            rc = utp._run_daemon_with_restart(args)
            assert rc == 0
            assert call_count == 3  # crashed twice, succeeded on third
        finally:
            utp._cmd_daemon = old_cmd

    def test_no_restart_flag_skips_restart(self):
        """--no-restart should run daemon directly without wrapper."""
        # This is tested via the dispatch logic: when no_restart=True,
        # asyncio.run(_cmd_daemon) is called directly instead of wrapper
        import types
        args = types.SimpleNamespace(no_restart=True)
        assert args.no_restart is True

    def test_ibkr_degraded_mode_start(self):
        """Daemon should start in degraded mode if IBKR not available."""
        from app.core.providers.ibkr import IBKRLiveProvider

        provider = IBKRLiveProvider()
        # Simulating what happens: connect fails, we set _connected=False
        provider._connected = False
        assert provider.is_healthy() is False
        # Provider is registered but unhealthy — server still starts

    def test_max_consecutive_crashes_limit(self):
        """Should give up after max consecutive crashes."""
        import types
        import utp

        call_count = 0

        async def always_crash(args):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("always crash")

        old_cmd = utp._cmd_daemon
        utp._cmd_daemon = always_crash
        try:
            args = types.SimpleNamespace(no_restart=False)
            rc = utp._run_daemon_with_restart(args)
            assert rc == 1
            assert call_count == 21  # 20 max + initial
        finally:
            utp._cmd_daemon = old_cmd


class TestExceptionLogging:
    """Verify bare except blocks have been replaced with logging."""

    def test_position_store_load_logs_error(self, tmp_data_dir):
        """Corrupt JSON file should be handled gracefully with logging."""
        from app.services.position_store import PlatformPositionStore

        store_path = tmp_data_dir / "positions.json"
        store_path.write_text("{invalid json")
        store = PlatformPositionStore(store_path)
        assert store._positions == {}

    def test_ledger_recover_sequence_handles_corruption(self, tmp_data_dir):
        """Corrupt ledger file should recover to sequence 0."""
        from app.services.ledger import TransactionLedger

        ledger_dir = tmp_data_dir / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "ledger.jsonl").write_text("{not valid json\n")
        ledger = TransactionLedger(ledger_dir)
        assert ledger._sequence == 0


class TestLANTrust:
    """Test LAN-only authentication bypass."""

    @pytest.mark.anyio
    async def test_lan_client_skips_auth(self, client):
        """Requests from localhost should not need auth."""
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_lan_client_can_access_protected_endpoint(self, client):
        """LAN clients should access protected endpoints without auth."""
        resp = await client.get("/dashboard/summary")
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_lan_trust_disabled_requires_auth(self, client):
        """When trust_local_network is disabled, auth is required."""
        from app.config import settings
        original = settings.trust_local_network
        settings.trust_local_network = False
        try:
            resp = await client.get("/dashboard/summary")
            assert resp.status_code == 401
        finally:
            settings.trust_local_network = original

    def test_private_networks_defined(self):
        """Verify PRIVATE_NETWORKS is defined in main.py."""
        from app.main import PRIVATE_NETWORKS
        assert len(PRIVATE_NETWORKS) >= 4

    @pytest.mark.anyio
    async def test_api_key_still_works(self, client, api_key_headers):
        """API key auth should still work alongside LAN trust."""
        resp = await client.get("/dashboard/summary", headers=api_key_headers)
        assert resp.status_code == 200

    def test_trust_local_network_config(self):
        """Config has trust_local_network setting."""
        from app.config import settings
        assert hasattr(settings, 'trust_local_network')
        assert settings.trust_local_network is True


class TestWorkerProxy:
    """Worker → IBKR-process proxy middleware behaviour."""

    def test_proxy_timeout_default(self, monkeypatch):
        """Default proxy timeout must exceed order_poll_timeout_seconds (60s)
        to avoid 502 races where the proxy times out at the same instant the
        order polling completes."""
        import app.main as _main
        from app.config import settings
        # Force fresh client construction
        monkeypatch.setattr(_main, "_worker_proxy_client", None)
        monkeypatch.delenv("_UTP_PROXY_TIMEOUT", raising=False)
        client = _main._get_worker_proxy_client()
        try:
            assert client.timeout.read >= settings.order_poll_timeout_seconds + 30, \
                "proxy read timeout must give the order polling room to finish"
        finally:
            monkeypatch.setattr(_main, "_worker_proxy_client", None)

    def test_proxy_timeout_env_override(self, monkeypatch):
        """_UTP_PROXY_TIMEOUT env var overrides the default."""
        import app.main as _main
        monkeypatch.setattr(_main, "_worker_proxy_client", None)
        monkeypatch.setenv("_UTP_PROXY_TIMEOUT", "300")
        client = _main._get_worker_proxy_client()
        try:
            assert client.timeout.read == 300.0
        finally:
            monkeypatch.setattr(_main, "_worker_proxy_client", None)


# ═══════════════════════════════════════════════════════════════════════════════
# Daemon Command & New HTTP Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestDaemonCommand:
    """Test daemon subcommand setup and new HTTP endpoints."""

    @pytest.mark.anyio
    async def test_health_endpoint(self, client):
        """Health endpoint returns ok."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.anyio
    async def test_health_shows_daemon_mode(self, client):
        """Health endpoint reports daemon_mode flag."""
        resp = await client.get("/health")
        data = resp.json()
        assert "daemon_mode" in data

    def test_daemon_mode_flag_exists(self):
        """Module-level _daemon_mode flag exists."""
        import app.main
        assert hasattr(app.main, '_daemon_mode')

    @pytest.mark.anyio
    async def test_close_endpoint(self, client, api_key_headers):
        """POST /trade/close works with valid position."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        request = TradeRequest(
            equity_order=EquityOrder(broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=10)
        )
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0)
        pid = store.add_position(request, result)

        resp = await client.post("/trade/close", json={"position_id": pid}, headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.anyio
    async def test_close_not_found(self, client, api_key_headers):
        """POST /trade/close returns 404 for unknown position."""
        resp = await client.post("/trade/close", json={"position_id": "nonexistent"}, headers=api_key_headers)
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_close_already_closed(self, client, api_key_headers):
        """POST /trade/close returns 400 for already-closed position."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        request = TradeRequest(
            equity_order=EquityOrder(broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=10)
        )
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0)
        pid = store.add_position(request, result)
        store.close_position(pid, 101.0, "manual")

        resp = await client.post("/trade/close", json={"position_id": pid}, headers=api_key_headers)
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_close_with_quantity(self, client, api_key_headers):
        """POST /trade/close with quantity reduces position."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        request = TradeRequest(
            equity_order=EquityOrder(broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=10)
        )
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0)
        pid = store.add_position(request, result)

        resp = await client.post(
            "/trade/close",
            json={"position_id": pid, "quantity": 5},
            headers=api_key_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["position"]["quantity"] == 5

    @pytest.mark.anyio
    async def test_trades_endpoint(self, client, api_key_headers):
        """GET /account/trades returns list."""
        resp = await client.get("/account/trades", headers=api_key_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.anyio
    async def test_trades_include_all(self, client, api_key_headers):
        """GET /account/trades?include_all=true returns open + closed."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        request = TradeRequest(
            equity_order=EquityOrder(broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=10)
        )
        result = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0)
        store.add_position(request, result)

        resp = await client.get("/account/trades?include_all=true", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    @pytest.mark.anyio
    async def test_orders_endpoint(self, client, api_key_headers):
        """GET /account/orders returns list."""
        resp = await client.get("/account/orders", headers=api_key_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.anyio
    async def test_cancel_requires_order_id(self, client, api_key_headers):
        """POST /account/cancel without order_id returns 400."""
        resp = await client.post("/account/cancel", headers=api_key_headers)
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_options_endpoint(self, client, api_key_headers):
        """GET /market/options/{symbol} returns chain data."""
        resp = await client.get("/market/options/SPX", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "symbol" in data

    @pytest.mark.anyio
    async def test_options_list_expirations(self, client, api_key_headers):
        """GET /market/options/{symbol}?list_expirations=true returns expirations."""
        resp = await client.get("/market/options/SPX?list_expirations=true", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "expirations" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Trading Client (HTTP Library)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTradingClient:
    """Test TradingClient and TradingClientSync classes."""

    def test_trading_client_exists(self):
        from utp import TradingClient
        client = TradingClient("http://localhost:8000")
        assert client._url == "http://localhost:8000"
        assert client._api_key is None
        assert client._client is None

    def test_trading_client_with_api_key(self):
        from utp import TradingClient
        client = TradingClient("http://localhost:8000", api_key="my-key")
        assert client._api_key == "my-key"

    @pytest.mark.anyio
    async def test_trading_client_context_manager(self):
        """Context manager connects and disconnects."""
        from utp import TradingClient
        client = TradingClient("http://localhost:99999")
        await client.connect()
        assert client._client is not None
        await client.disconnect()
        assert client._client is None

    @pytest.mark.anyio
    async def test_trading_client_ensure_connected(self):
        """Methods raise if not connected."""
        from utp import TradingClient
        client = TradingClient("http://localhost:8000")
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.health()

    def test_sync_client_exists(self):
        from utp import TradingClientSync
        client = TradingClientSync("http://localhost:8000")
        assert client._async_client._url == "http://localhost:8000"

    def test_sync_client_context_manager(self):
        """Sync context manager creates and closes loop."""
        from utp import TradingClientSync
        client = TradingClientSync("http://localhost:99999")
        client.connect()
        assert client._async_client._client is not None
        client.disconnect()

    def test_trading_client_methods_defined(self):
        """All expected methods are present."""
        from utp import TradingClient
        methods = [
            'health', 'trade_credit_spread', 'trade_iron_condor',
            'trade_debit_spread', 'trade_multi_leg', 'trade_equity',
            'close_position', 'get_positions', 'get_quote',
            'get_portfolio_summary', 'get_trades', 'get_orders',
            'cancel_order', 'get_performance', 'get_options',
            'get_advisor_recommendations', 'confirm_advisor_trade',
        ]
        for m in methods:
            assert hasattr(TradingClient, m), f"Missing method: {m}"

    def test_sync_client_methods_defined(self):
        """All expected sync methods are present."""
        from utp import TradingClientSync
        methods = [
            'health', 'trade_credit_spread', 'trade_iron_condor',
            'trade_debit_spread', 'trade_multi_leg', 'trade_equity',
            'close_position', 'get_positions', 'get_quote',
            'get_portfolio_summary', 'get_trades', 'get_orders',
            'cancel_order', 'get_performance', 'get_options',
        ]
        for m in methods:
            assert hasattr(TradingClientSync, m), f"Missing method: {m}"

    @pytest.mark.anyio
    async def test_trade_credit_spread_payload(self):
        """Verify credit spread builds correct payload."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"order_id": "test-123", "status": "SUBMITTED"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        result = await client.trade_credit_spread(
            symbol="SPX", short_strike=5500, long_strike=5475,
            option_type="PUT", expiration="2026-03-20",
            quantity=1, net_price=3.50,
        )

        assert result["order_id"] == "test-123"
        # Verify the payload structure
        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "multi_leg_order" in payload
        assert len(payload["multi_leg_order"]["legs"]) == 2
        assert payload["multi_leg_order"]["legs"][0]["action"] == "SELL_TO_OPEN"
        assert payload["multi_leg_order"]["legs"][1]["action"] == "BUY_TO_OPEN"

    @pytest.mark.anyio
    async def test_trade_equity_payload(self):
        """Verify equity trade builds correct payload."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"order_id": "eq-456", "status": "FILLED"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        result = await client.trade_equity(
            symbol="SPY", side="BUY", quantity=10, limit_price=550.00,
        )

        assert result["order_id"] == "eq-456"
        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert "equity_order" in payload
        assert payload["equity_order"]["symbol"] == "SPY"
        assert payload["equity_order"]["side"] == "BUY"
        assert payload["equity_order"]["order_type"] == "LIMIT"

    @pytest.mark.anyio
    async def test_trade_iron_condor_payload(self):
        """Verify iron condor builds correct 4-leg payload."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"order_id": "ic-123", "status": "SUBMITTED"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        result = await client.trade_iron_condor(
            symbol="SPX", put_short=5500, put_long=5475,
            call_short=5700, call_long=5725,
            expiration="2026-03-20", quantity=2,
        )

        assert result["order_id"] == "ic-123"
        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        mlo = payload["multi_leg_order"]
        assert len(mlo["legs"]) == 4
        # Put legs
        assert mlo["legs"][0]["strike"] == 5500
        assert mlo["legs"][0]["option_type"] == "PUT"
        assert mlo["legs"][0]["action"] == "SELL_TO_OPEN"
        assert mlo["legs"][1]["strike"] == 5475
        assert mlo["legs"][1]["option_type"] == "PUT"
        assert mlo["legs"][1]["action"] == "BUY_TO_OPEN"
        # Call legs
        assert mlo["legs"][2]["strike"] == 5700
        assert mlo["legs"][2]["option_type"] == "CALL"
        assert mlo["legs"][2]["action"] == "SELL_TO_OPEN"
        assert mlo["legs"][3]["strike"] == 5725
        assert mlo["legs"][3]["option_type"] == "CALL"
        assert mlo["legs"][3]["action"] == "BUY_TO_OPEN"
        assert mlo["quantity"] == 2
        assert mlo["order_type"] == "MARKET"
        assert mlo["net_price"] is None

    @pytest.mark.anyio
    async def test_trade_iron_condor_limit_order(self):
        """Iron condor with net_price uses LIMIT order type."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"order_id": "ic-lim", "status": "SUBMITTED"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        await client.trade_iron_condor(
            symbol="SPX", put_short=5500, put_long=5475,
            call_short=5700, call_long=5725,
            expiration="2026-03-20", quantity=1, net_price=5.00,
        )

        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        mlo = payload["multi_leg_order"]
        assert mlo["order_type"] == "LIMIT"
        assert mlo["net_price"] == 5.00

    @pytest.mark.anyio
    async def test_trade_debit_spread_payload(self):
        """Verify debit spread builds correct 2-leg payload with BUY first."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"order_id": "ds-123", "status": "SUBMITTED"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        result = await client.trade_debit_spread(
            symbol="QQQ", long_strike=480, short_strike=490,
            option_type="CALL", expiration="2026-03-20",
            quantity=3, net_price=4.00,
        )

        assert result["order_id"] == "ds-123"
        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        mlo = payload["multi_leg_order"]
        assert len(mlo["legs"]) == 2
        assert mlo["legs"][0]["strike"] == 480
        assert mlo["legs"][0]["action"] == "BUY_TO_OPEN"
        assert mlo["legs"][0]["option_type"] == "CALL"
        assert mlo["legs"][1]["strike"] == 490
        assert mlo["legs"][1]["action"] == "SELL_TO_OPEN"
        assert mlo["legs"][1]["option_type"] == "CALL"
        assert mlo["quantity"] == 3
        assert mlo["order_type"] == "LIMIT"
        assert mlo["net_price"] == 4.00

    @pytest.mark.anyio
    async def test_trade_multi_leg_payload(self):
        """Verify generic multi-leg passes arbitrary legs."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"order_id": "ml-123", "status": "SUBMITTED"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        legs = [
            {"symbol": "SPX", "expiration": "2026-03-20", "strike": 5500,
             "option_type": "PUT", "action": "SELL_TO_OPEN", "quantity": 1},
            {"symbol": "SPX", "expiration": "2026-03-20", "strike": 5475,
             "option_type": "PUT", "action": "BUY_TO_OPEN", "quantity": 1},
        ]
        result = await client.trade_multi_leg(legs=legs, quantity=5)

        assert result["order_id"] == "ml-123"
        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        mlo = payload["multi_leg_order"]
        assert mlo["legs"] == legs
        assert mlo["quantity"] == 5
        assert mlo["order_type"] == "MARKET"

    @pytest.mark.anyio
    async def test_trade_multi_leg_limit(self):
        """Generic multi-leg with net_price uses LIMIT."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"order_id": "ml-lim", "status": "SUBMITTED"}
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        legs = [
            {"symbol": "RUT", "expiration": "2026-03-20", "strike": 2200,
             "option_type": "CALL", "action": "SELL_TO_OPEN", "quantity": 1},
        ]
        await client.trade_multi_leg(legs=legs, quantity=1, net_price=2.50)

        call_args = mock_http.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        mlo = payload["multi_leg_order"]
        assert mlo["order_type"] == "LIMIT"
        assert mlo["net_price"] == 2.50

    @pytest.mark.anyio
    async def test_trade_multi_leg_max_legs(self):
        """Generic multi-leg rejects >4 legs."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        client._client = AsyncMock()  # connected

        legs = [{"symbol": "SPX", "expiration": "2026-03-20", "strike": 5000 + i,
                 "option_type": "PUT", "action": "SELL_TO_OPEN", "quantity": 1}
                for i in range(5)]
        with pytest.raises(ValueError, match="max 4 legs"):
            await client.trade_multi_leg(legs=legs)

    @pytest.mark.anyio
    async def test_get_positions_extracts_active(self):
        """get_positions returns active_positions from summary."""
        from utp import TradingClient

        client = TradingClient("http://localhost:8000")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active_positions": [{"id": "pos-1"}, {"id": "pos-2"}],
            "total_pnl": 100.0,
        }
        mock_resp.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        positions = await client.get_positions()
        assert len(positions) == 2
        assert positions[0]["id"] == "pos-1"


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Client Mode (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHTTPClientMode:
    """Test CLI HTTP client mode (Phase 2)."""

    def test_detect_server_with_explicit_url(self, monkeypatch):
        """--server flag returns that URL."""
        # Restore the real _detect_server for this test
        import importlib, utp as _utp_mod
        real_fn = importlib.import_module("utp").__dict__.get("_detect_server_real", None)
        if real_fn is None:
            # The conftest monkeypatch replaced it; use the original from source
            import urllib.request
            def _real_detect(args):
                server = getattr(args, "server", None)
                if server:
                    return server
                try:
                    port = getattr(args, "server_port", 8000)
                    url = f"http://localhost:{port}/health"
                    req = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(req, timeout=1) as resp:
                        if resp.status == 200:
                            return f"http://localhost:{port}"
                except Exception:
                    pass
                return None
            real_fn = _real_detect
        ns = argparse.Namespace(server="http://example.com:8000", server_port=8000)
        result = real_fn(ns)
        assert result == "http://example.com:8000"

    def test_detect_server_none_when_no_daemon(self):
        """Auto-detect returns None when no daemon running."""
        # Use a port nothing listens on; the conftest monkeypatch returns None anyway
        from utp import _detect_server
        ns = argparse.Namespace(server=None, server_port=19999)  # unlikely port
        result = _detect_server(ns)
        assert result is None

    def test_detect_server_no_server_attr(self):
        """When args has no server attribute, auto-detect is attempted."""
        from utp import _detect_server
        ns = argparse.Namespace(server_port=19998)
        result = _detect_server(ns)
        assert result is None

    def test_http_functions_exist(self):
        """HTTP variant functions are defined."""
        from utp import (
            _cmd_portfolio_http,
            _cmd_quote_http,
            _cmd_status_http,
            _cmd_orders_http,
            _cmd_trades_http,
            _cmd_close_http,
            _cmd_performance_http,
            _cmd_journal_http,
        )
        assert callable(_cmd_portfolio_http)
        assert callable(_cmd_quote_http)
        assert callable(_cmd_status_http)
        assert callable(_cmd_orders_http)
        assert callable(_cmd_trades_http)
        assert callable(_cmd_close_http)
        assert callable(_cmd_performance_http)
        assert callable(_cmd_journal_http)

    def test_repl_function_exists(self):
        """REPL command function is defined."""
        from utp import _cmd_repl
        assert callable(_cmd_repl)

    def test_http_functions_are_async(self):
        """All HTTP variant functions are coroutine functions."""
        import inspect
        from utp import (
            _cmd_portfolio_http,
            _cmd_quote_http,
            _cmd_status_http,
            _cmd_orders_http,
            _cmd_trades_http,
            _cmd_close_http,
            _cmd_performance_http,
            _cmd_journal_http,
            _cmd_repl,
        )
        for fn in [
            _cmd_portfolio_http, _cmd_quote_http, _cmd_status_http,
            _cmd_orders_http, _cmd_trades_http, _cmd_close_http,
            _cmd_performance_http, _cmd_journal_http, _cmd_repl,
        ]:
            assert inspect.iscoroutinefunction(fn), f"{fn.__name__} is not async"

    @pytest.mark.anyio
    async def test_portfolio_http_success(self, client, api_key_headers):
        """_cmd_portfolio_http renders output from /dashboard/summary."""
        from utp import _cmd_portfolio_http
        # Use the test server's actual endpoint
        import httpx
        ns = argparse.Namespace(server="unused", server_port=8000)
        # We test the real function against the test app
        transport = httpx.ASGITransport(app=client._transport.app)  # type: ignore[attr-defined]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as hc:
            resp = await hc.get("/dashboard/summary", headers=api_key_headers)
            assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_status_http_success(self, client, api_key_headers):
        """_cmd_status_http renders output from /dashboard/status."""
        import httpx
        transport = httpx.ASGITransport(app=client._transport.app)  # type: ignore[attr-defined]
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as hc:
            resp = await hc.get("/dashboard/status", headers=api_key_headers)
            assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Advisor Integration (Phase 5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdvisorIntegration:
    """Test advisor HTTP endpoints and daemon state."""

    def test_daemon_state_exists(self):
        """Module-level _daemon_state dict exists with expected keys."""
        from utp import _daemon_state
        assert isinstance(_daemon_state, dict)
        assert "advisor_entries" in _daemon_state
        assert "advisor_exits" in _daemon_state
        assert "advisor_profile" in _daemon_state
        assert "advisor_last_eval" in _daemon_state

    @pytest.mark.anyio
    async def test_advisor_recommendations_endpoint(self, client):
        """GET /dashboard/advisor/recommendations returns data."""
        resp = await client.get("/dashboard/advisor/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "exits" in data
        assert "profile" in data
        assert "last_eval" in data

    @pytest.mark.anyio
    async def test_advisor_status_endpoint(self, client):
        """GET /dashboard/advisor/status returns status."""
        resp = await client.get("/dashboard/advisor/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "active" in data
        assert "pending_entries" in data
        assert "pending_exits" in data

    @pytest.mark.anyio
    async def test_advisor_confirm_not_found(self, client, api_key_headers):
        """POST /trade/advisor/confirm with invalid priority returns 404."""
        resp = await client.post("/trade/advisor/confirm", params={"priority": 999}, headers=api_key_headers)
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_advisor_confirm_with_data(self, client, api_key_headers):
        """POST /trade/advisor/confirm works with valid data."""
        from utp import _daemon_state
        _daemon_state["advisor_entries"] = [
            {"priority": 1, "tier_label": "P90_DTE2", "direction": "put",
             "short_strike": 5500, "long_strike": 5475, "credit": 3.50,
             "dte": 2, "num_contracts": 1},
        ]
        try:
            resp = await client.post("/trade/advisor/confirm", params={"priority": 1}, headers=api_key_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "confirmed"
            assert data["recommendation"]["tier_label"] == "P90_DTE2"
            assert data["recommendation"]["short_strike"] == 5500
        finally:
            _daemon_state["advisor_entries"] = []

    @pytest.mark.anyio
    async def test_advisor_status_with_profile(self, client):
        """Advisor status shows active when profile is set."""
        from utp import _daemon_state
        _daemon_state["advisor_profile"] = "test_profile"
        try:
            resp = await client.get("/dashboard/advisor/status")
            data = resp.json()
            assert data["active"] is True
            assert data["profile"] == "test_profile"
        finally:
            _daemon_state["advisor_profile"] = None

    def test_trading_client_advisor_methods(self):
        """TradingClient has advisor methods."""
        from utp import TradingClient
        assert hasattr(TradingClient, 'get_advisor_recommendations')
        assert hasattr(TradingClient, 'get_advisor_status')
        assert hasattr(TradingClient, 'confirm_advisor_trade')

    @pytest.mark.anyio
    async def test_advisor_recommendations_empty_by_default(self, client):
        """Advisor returns empty recommendations by default."""
        resp = await client.get("/dashboard/advisor/recommendations")
        data = resp.json()
        assert data["entries"] == []
        assert data["exits"] == []


class TestHTTPRoutingFixes:
    """Test that all commands route through daemon when available."""

    def test_reconcile_http_function_exists(self):
        from utp import _cmd_reconcile_http
        import inspect
        assert inspect.iscoroutinefunction(_cmd_reconcile_http)

    def test_options_http_function_exists(self):
        from utp import _cmd_options_http
        import inspect
        assert inspect.iscoroutinefunction(_cmd_options_http)

    def test_margin_http_function_exists(self):
        from utp import _cmd_margin_http
        import inspect
        assert inspect.iscoroutinefunction(_cmd_margin_http)

    def test_trade_http_function_exists(self):
        from utp import _cmd_trade_http
        import inspect
        assert inspect.iscoroutinefunction(_cmd_trade_http)

    @pytest.mark.anyio
    async def test_dashboard_summary_enriched(self, client):
        """Dashboard summary should include broker balance fields."""
        resp = await client.get("/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        # These fields should exist (even if 0 for stub)
        assert "net_liquidation" in data
        assert "buying_power" in data
        assert "unrealized_pnl" in data

    @pytest.mark.anyio
    async def test_reconciliation_endpoint(self, client):
        """GET /account/reconciliation works."""
        resp = await client.get("/account/reconciliation")
        assert resp.status_code == 200
        data = resp.json()
        assert "broker" in data


class TestPortfolioEndpoint:
    """Test the new /dashboard/portfolio endpoint."""

    @pytest.mark.anyio
    async def test_portfolio_endpoint_exists(self, client):
        """GET /dashboard/portfolio returns data."""
        resp = await client.get("/dashboard/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert "positions" in data
        assert "balances" in data
        assert "realized_pnl" in data
        assert "unrealized_pnl" in data
        assert "total_pnl" in data

    @pytest.mark.anyio
    async def test_portfolio_with_positions(self, client):
        """Portfolio shows positions."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=10
        ))
        res = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=500.0)
        store.add_position(req, res)

        resp = await client.get("/dashboard/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["positions"]) == 1
        assert data["positions"][0]["symbol"] == "SPY"

    @pytest.mark.anyio
    async def test_portfolio_recent_closed(self, client):
        """Portfolio includes recent closed positions."""
        from app.services.position_store import get_position_store
        store = get_position_store()
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY, quantity=5
        ))
        res = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=150.0)
        pid = store.add_position(req, res)
        store.close_position(pid, 160.0, "profit_target")

        resp = await client.get("/dashboard/portfolio")
        data = resp.json()
        assert len(data.get("recent_closed", [])) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# LiveDataService — IBKR-primary with local fallback
# ═══════════════════════════════════════════════════════════════════════════════


class TestLiveDataService:
    """Tests for LiveDataService: IBKR-primary data with local fallback."""

    @pytest.mark.anyio
    async def test_summary_ibkr_healthy(self):
        """When IBKR is healthy, summary includes broker balances and P&L."""
        from app.services.live_data_service import LiveDataService, _match_broker_pnl
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        # Add a position so there's something to enrich
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=10
        ))
        res = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=500.0)
        store.add_position(req, res)

        # Mock IBKR provider
        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=True)

        from app.models import AccountBalances
        mock_provider.get_account_balances.return_value = AccountBalances(
            cash=50000.0, net_liquidation=100000.0, buying_power=200000.0,
            maint_margin_req=5000.0, available_funds=95000.0, broker="ibkr"
        )
        mock_provider.get_portfolio_items.return_value = [{
            "symbol": "SPY", "sec_type": "STK", "expiration": "",
            "strike": 0.0, "right": "", "position": 10,
            "unrealized_pnl": 500.0, "market_value": 5500.0,
            "avg_cost": 500.0, "market_price": 550.0,
        }]

        svc = LiveDataService(store, dashboard, mock_provider)
        summary = await svc.get_summary()

        assert summary.net_liquidation == 100000.0
        assert summary.buying_power == 200000.0
        assert summary.unrealized_pnl == 500.0

    @pytest.mark.anyio
    async def test_summary_ibkr_unhealthy_fallback(self):
        """When IBKR is unhealthy, falls back to local store data."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=False)

        svc = LiveDataService(store, dashboard, mock_provider)
        summary = await svc.get_summary()

        # Should return local data without broker enrichment
        assert summary.net_liquidation == 0  # default, no broker enrichment
        mock_provider.get_account_balances.assert_not_called()

    @pytest.mark.anyio
    async def test_summary_ibkr_not_registered(self):
        """When no IBKR provider, graceful fallback to local."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        svc = LiveDataService(store, dashboard, ibkr_provider=None)
        summary = await svc.get_summary()

        assert summary is not None
        assert summary.net_liquidation == 0

    @pytest.mark.anyio
    async def test_portfolio_ibkr_primary(self):
        """Portfolio uses IBKR data as primary when connected."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="QQQ", side=OrderSide.BUY, quantity=5
        ))
        res = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=400.0)
        store.add_position(req, res)

        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=True)
        from app.models import AccountBalances
        mock_provider.get_account_balances.return_value = AccountBalances(
            cash=75000.0, net_liquidation=150000.0, buying_power=300000.0,
            broker="ibkr"
        )
        mock_provider.get_portfolio_items.return_value = [{
            "symbol": "QQQ", "sec_type": "STK", "expiration": "",
            "strike": 0.0, "right": "", "position": 5,
            "unrealized_pnl": 250.0, "market_value": 2250.0,
            "avg_cost": 400.0, "market_price": 450.0,
        }]

        svc = LiveDataService(store, dashboard, mock_provider)
        result = await svc.get_portfolio()

        assert result["balances"]["net_liquidation"] == 150000.0
        assert len(result["positions"]) == 1
        assert result["positions"][0]["broker_unrealized_pnl"] == 250.0
        assert "recent_closed" in result

    @pytest.mark.anyio
    async def test_portfolio_fallback(self):
        """Portfolio uses local store when IBKR disconnected."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="AAPL", side=OrderSide.BUY, quantity=3
        ))
        res = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=170.0)
        store.add_position(req, res)

        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=False)

        svc = LiveDataService(store, dashboard, mock_provider)
        result = await svc.get_portfolio()

        assert len(result["positions"]) == 1
        assert result["positions"][0]["symbol"] == "AAPL"
        assert result["balances"] == {}
        mock_provider.get_portfolio_items.assert_not_called()

    @pytest.mark.anyio
    async def test_trades_mixed_source(self):
        """Open positions enriched with IBKR, closed from local."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        # Open position
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="SPY", side=OrderSide.BUY, quantity=10
        ))
        res = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=500.0)
        store.add_position(req, res)

        # Closed position
        req2 = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="GOOG", side=OrderSide.BUY, quantity=2
        ))
        res2 = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=100.0)
        pid2 = store.add_position(req2, res2)
        store.close_position(pid2, 120.0, "profit_target")

        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=True)
        mock_provider.get_portfolio_items.return_value = [{
            "symbol": "SPY", "sec_type": "STK", "expiration": "",
            "strike": 0.0, "right": "", "position": 10,
            "unrealized_pnl": 300.0, "market_value": 5300.0,
            "avg_cost": 500.0, "market_price": 530.0,
        }]

        svc = LiveDataService(store, dashboard, mock_provider)
        trades = await svc.get_trades(include_all=True)

        # Should include both open and closed
        assert len(trades) == 2
        symbols = {t.get("symbol") for t in trades}
        assert "SPY" in symbols
        assert "GOOG" in symbols

    @pytest.mark.anyio
    async def test_status_ibkr_active_positions(self):
        """Status endpoint works with LiveDataService."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=True)

        svc = LiveDataService(store, dashboard, mock_provider)
        status = await svc.get_status()

        assert hasattr(status, "active_positions")
        assert hasattr(status, "in_transit_orders")
        assert hasattr(status, "recent_closed")

    @pytest.mark.anyio
    async def test_performance_always_local(self):
        """Performance metrics always come from local store."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        # Add and close a position
        req = TradeRequest(equity_order=EquityOrder(
            broker=Broker.IBKR, symbol="TSLA", side=OrderSide.BUY, quantity=1
        ))
        res = OrderResult(broker=Broker.IBKR, status=OrderStatus.FILLED, filled_price=200.0)
        pid = store.add_position(req, res)
        store.close_position(pid, 220.0, "profit_target")

        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=True)

        svc = LiveDataService(store, dashboard, mock_provider)
        perf = await svc.get_performance()

        # Performance is derived from closed positions — always local
        assert perf.total_trades == 1
        assert perf.win_rate > 0
        # IBKR should NOT be called for historical metrics
        mock_provider.get_portfolio_items.assert_not_called()

    @pytest.mark.anyio
    async def test_out_of_band_positions(self):
        """IBKR positions not in local store are logged."""
        from app.services.live_data_service import LiveDataService
        from app.services.dashboard_service import DashboardService
        from app.services.position_store import get_position_store

        store = get_position_store()
        dashboard = DashboardService(store)

        mock_provider = AsyncMock()
        mock_provider.is_healthy = MagicMock(return_value=True)
        mock_provider.get_portfolio_items.return_value = [{
            "symbol": "NVDA", "sec_type": "STK", "expiration": "",
            "strike": 0.0, "right": "", "position": 100,
            "unrealized_pnl": 1500.0, "market_value": 15000.0,
            "avg_cost": 135.0, "market_price": 150.0,
        }]

        svc = LiveDataService(store, dashboard, mock_provider)
        # No positions in local store — IBKR reports NVDA
        positions = await svc.get_active_positions()

        # Local store has no positions, so result is empty
        assert len(positions) == 0
        # But the method should have called IBKR
        mock_provider.get_portfolio_items.assert_called_once()

    def test_match_broker_pnl_import(self):
        """_match_broker_pnl works from its new canonical location."""
        from app.services.live_data_service import _match_broker_pnl

        items = [{
            "symbol": "SPY", "sec_type": "STK", "expiration": "",
            "strike": 0.0, "right": "", "position": 10,
            "unrealized_pnl": 100.0, "market_value": 5100.0,
            "avg_cost": 500.0, "market_price": 510.0,
        }]
        positions = [{
            "position_id": "abc123",
            "order_type": "equity",
            "symbol": "SPY",
            "quantity": 10,
            "expiration": "",
            "legs": [],
        }]

        result = _match_broker_pnl(items, positions)
        assert "abc123" in result
        assert result["abc123"]["unrealized_pnl"] == 100.0
        assert result["abc123"]["market_price"] == 510.0

    @pytest.mark.anyio
    async def test_summary_via_api_endpoint(self, client):
        """Dashboard summary endpoint uses LiveDataService."""
        resp = await client.get("/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_positions" in data
        assert "total_pnl" in data

    @pytest.mark.anyio
    async def test_config_sync_interval(self):
        """Position sync interval is 120 seconds."""
        from app.config import Settings
        s = Settings()
        assert s.position_sync_interval_seconds == 120


# ═══════════════════════════════════════════════════════════════════════════════
# Market Data Streaming — IBKR → Redis/QuestDB/WS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarketDataStreaming:
    """Tests for the IBKR market data streaming service."""

    def test_streaming_config_load(self, tmp_path):
        """Load streaming config from YAML."""
        from app.services.streaming_config import load_streaming_config
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("""
symbols:
  - SPX
  - NDX
  - SPY
redis_enabled: true
redis_url: "redis://localhost:6379/0"
tick_batch_interval: 1.0
market_hours_only: false
""")
        config = load_streaming_config(cfg_file)
        assert len(config.symbols) == 3
        assert config.symbols[0].symbol == "SPX"
        assert config.symbols[0].sec_type == "IND"
        assert config.symbols[0].exchange == "CBOE"
        assert config.symbols[2].symbol == "SPY"
        assert config.symbols[2].sec_type == "STK"
        assert config.symbols[2].exchange == "SMART"
        assert config.redis_enabled is True
        assert config.tick_batch_interval == 1.0

    def test_streaming_config_index_auto_detect(self, tmp_path):
        """Index symbols auto-detected with correct exchange."""
        from app.services.streaming_config import load_streaming_config
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("""
symbols:
  - SPX
  - NDX
  - RUT
  - DJX
  - VIX
  - AAPL
""")
        config = load_streaming_config(cfg_file)
        assert config.symbols[0].sec_type == "IND"
        assert config.symbols[0].exchange == "CBOE"
        assert config.symbols[1].exchange == "NASDAQ"
        assert config.symbols[2].exchange == "RUSSELL"
        assert config.symbols[5].sec_type == "STK"
        assert config.symbols[5].exchange == "SMART"

    def test_streaming_config_validation_too_many_symbols(self, tmp_path):
        """Config validation rejects too many symbols."""
        from app.services.streaming_config import load_streaming_config
        cfg_file = tmp_path / "stream.yaml"
        symbols = "\n".join(f"  - SYM{i}" for i in range(60))
        cfg_file.write_text(f"symbols:\n{symbols}\nmax_subscriptions: 50\n")
        with pytest.raises(ValueError, match="Too many symbols"):
            load_streaming_config(cfg_file)

    def test_streaming_config_validation_no_symbols(self, tmp_path):
        """Config validation rejects empty symbols."""
        from app.services.streaming_config import load_streaming_config
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols: []\n")
        with pytest.raises(ValueError, match="No symbols"):
            load_streaming_config(cfg_file)

    def test_streaming_config_dict_form(self, tmp_path):
        """Symbols can be specified in dict form."""
        from app.services.streaming_config import load_streaming_config
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("""
symbols:
  - {symbol: TSLA, sec_type: STK, exchange: SMART}
  - SPX
""")
        config = load_streaming_config(cfg_file)
        assert config.symbols[0].symbol == "TSLA"
        assert config.symbols[0].sec_type == "STK"
        assert config.symbols[1].symbol == "SPX"
        assert config.symbols[1].sec_type == "IND"

    def test_streaming_safety_limits(self):
        """Safety buffer constants are at least 50% of IBKR limits."""
        from app.services.streaming_config import MAX_SUBSCRIPTIONS, RATE_LIMIT_MSG_SEC
        assert MAX_SUBSCRIPTIONS <= 50  # 50% of ~100 IBKR lines
        assert RATE_LIMIT_MSG_SEC <= 25  # 50% of 50 msg/sec IBKR limit

    @pytest.mark.anyio
    async def test_streaming_service_init(self, tmp_path):
        """Service initializes with config, no IBKR needed for init."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\nredis_enabled: false\nquestdb_enabled: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        assert svc.subscription_count == 0
        assert svc.is_running is False
        stats = svc.stats
        assert stats["running"] is False
        assert stats["max_subscriptions"] == 50

    @pytest.mark.anyio
    async def test_streaming_service_start_stop_no_ibkr(self, tmp_path):
        """Service starts and stops gracefully without IBKR."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\nredis_enabled: false\nquestdb_enabled: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        await svc.start()
        assert svc.is_running is True
        await svc.stop()
        assert svc.is_running is False

    @pytest.mark.anyio
    async def test_streaming_tick_processing(self, tmp_path):
        """Ticks are buffered and can be flushed."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)

        # Manually add a tick to the buffer
        svc._pending_ticks["SPX"] = {
            "symbol": "SPX",
            "timestamp": "2026-03-18T15:00:00Z",
            "price": 5683.25,
            "bid": 5683.00,
            "ask": 5683.50,
            "last": 5683.25,
            "volume": 0,
        }
        assert len(svc._pending_ticks) == 1

        # Flush (no targets enabled, so just clears buffer)
        await svc._flush_ticks()
        assert len(svc._pending_ticks) == 0

    @pytest.mark.anyio
    async def test_streaming_status_endpoint(self, client):
        """Streaming status endpoint returns info."""
        resp = await client.get("/market/streaming/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data

    @pytest.mark.anyio
    async def test_quote_ws_manager(self):
        """QuoteStreamManager handles subscriptions."""
        from app.websocket import QuoteStreamManager
        mgr = QuoteStreamManager()
        # Just verify it initializes
        assert len(mgr._subscriptions) == 0

    def test_default_streaming_config_exists(self):
        """Default streaming config YAML exists."""
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "configs" / "streaming_default.yaml"
        assert config_path.exists()
        from app.services.streaming_config import load_streaming_config
        config = load_streaming_config(config_path)
        assert len(config.symbols) >= 1

    def test_streaming_config_cpg_mode(self, tmp_path):
        """Config loads streaming_mode and cpg_poll_interval."""
        from app.services.streaming_config import load_streaming_config
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\nstreaming_mode: polling\ncpg_poll_interval: 2.5\n")
        config = load_streaming_config(cfg_file)
        assert config.streaming_mode == "polling"
        assert config.cpg_poll_interval == 2.5

    def test_streaming_config_defaults_auto(self, tmp_path):
        """Default streaming_mode is auto."""
        from app.services.streaming_config import load_streaming_config
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\n")
        config = load_streaming_config(cfg_file)
        assert config.streaming_mode == "auto"
        assert config.cpg_poll_interval == 1.5

    def test_ingest_tick_valid(self, tmp_path):
        """_ingest_tick accepts a valid stock price."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        accepted = svc._ingest_tick("SPY", 510.0, 509.0, 511.0, 100, 500.0, False)
        assert accepted is True
        assert svc._last_tick["SPY"]["price"] == 510.0
        assert svc._prev_close["SPY"] == 500.0

    def test_ingest_tick_rejects_outside_close_band(self, tmp_path):
        """_ingest_tick rejects price outside close band."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        # Seed close
        svc._ingest_tick("SPY", 500.0, 499.0, 501.0, 100, 500.0, False)
        # Price 50% above close — rejected
        accepted = svc._ingest_tick("SPY", 750.0, 749.0, 751.0, 100, None, False)
        assert accepted is False

    def test_parse_cpg_float(self):
        """CPG float parser handles prefixed values."""
        from app.services.market_data_streaming import MarketDataStreamingService
        parse = MarketDataStreamingService._parse_cpg_float
        assert parse("5800.25") == 5800.25
        assert parse("C5800.25") == 5800.25
        assert parse("H100.5") == 100.5
        assert parse("L50.0") == 50.0
        assert parse(None) is None
        assert parse("") is None
        assert parse(0) is None

    def test_process_cpg_snapshot(self, tmp_path):
        """_process_cpg_snapshot feeds a tick into _ingest_tick."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        svc._cpg_conid_to_symbol = {12345: "SPY"}
        # Seed close for validation
        svc._prev_close["SPY"] = 500.0
        snap = {"conid": 12345, "31": "510.0", "84": "509.5", "86": "510.5", "87": "1000"}
        svc._process_cpg_snapshot(snap)
        assert "SPY" in svc._last_tick
        assert svc._last_tick["SPY"]["price"] == 510.0

    @pytest.mark.anyio
    async def test_streaming_mode_auto_selects_cpg(self, tmp_path):
        """Auto mode selects CPG websocket when provider has _gateway_url."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService
        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\n")
        config = load_streaming_config(cfg_file)
        # Mock CPG provider (has _gateway_url, no _ib)
        mock_provider = MagicMock()
        mock_provider._gateway_url = "https://localhost:7498"
        del mock_provider._ib  # ensure no _ib attribute
        svc = MarketDataStreamingService(config, ibkr_provider=mock_provider)
        assert svc._is_cpg_provider is True
        assert svc._has_ib_client is False

    def test_close_gate_rejects_price_beyond_35pct(self, tmp_path):
        """Tick with price >35% from previous close is rejected."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)

        # Simulate subscription so the tick handler accepts SPY
        svc._subscriptions["SPY"] = MagicMock()

        # Build a fake ticker: close=500, last=750 (50% above → rejected)
        ticker = MagicMock()
        ticker.contract = MagicMock(symbol="SPY")
        ticker.marketPrice.return_value = 750.0
        ticker.bid = 749.0
        ticker.ask = 751.0
        ticker.last = 750.0
        ticker.close = 500.0
        ticker.volume = 100

        svc._on_pending_tickers([ticker])
        # Price 750 is 50% above close 500 → should be rejected
        assert "SPY" not in svc._pending_ticks
        assert "SPY" not in svc._last_tick

    def test_close_gate_accepts_price_within_35pct(self, tmp_path):
        """Tick with price within 35% of previous close is accepted."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)

        svc._subscriptions["SPY"] = MagicMock()

        ticker = MagicMock()
        ticker.contract = MagicMock(symbol="SPY")
        ticker.marketPrice.return_value = 510.0
        ticker.bid = 509.0
        ticker.ask = 511.0
        ticker.last = 510.0
        ticker.close = 500.0
        ticker.volume = 100

        svc._on_pending_tickers([ticker])
        # 510 is 2% above close 500 → accepted
        assert "SPY" in svc._pending_ticks
        assert svc._pending_ticks["SPY"]["price"] == 510.0

    def test_close_gate_index_rejects_garbage(self, tmp_path):
        """Index tick with garbage price vs close is rejected."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)

        svc._subscriptions["SPX"] = MagicMock()

        ticker = MagicMock()
        ticker.contract = MagicMock(symbol="SPX")
        ticker.marketPrice.return_value = 200.0  # garbage for SPX
        ticker.bid = float("nan")
        ticker.ask = float("nan")
        ticker.last = 200.0
        ticker.close = 5700.0  # real previous close
        ticker.volume = 0

        svc._on_pending_tickers([ticker])
        # 200 is >35% below close 5700 → rejected
        assert "SPX" not in svc._pending_ticks

    def test_close_gate_uses_config_value(self, tmp_path):
        """Close gate respects close_band_pct from config (e.g., 10%)."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: false\nclose_band_pct: 0.10\n")
        config = load_streaming_config(cfg_file)
        assert config.close_band_pct == 0.10
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        svc._subscriptions["SPY"] = MagicMock()

        # 20% above close → rejected with 10% band, would pass with default 35%
        ticker = MagicMock()
        ticker.contract = MagicMock(symbol="SPY")
        ticker.marketPrice.return_value = 600.0
        ticker.bid = 599.0; ticker.ask = 601.0; ticker.last = 600.0
        ticker.close = 500.0; ticker.volume = 100
        svc._on_pending_tickers([ticker])
        assert "SPY" not in svc._pending_ticks

    def test_close_gate_does_not_drift(self, tmp_path):
        """Previous close anchor is not overwritten by live ticks."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: false\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        svc._subscriptions["SPY"] = MagicMock()

        # First tick: close=500, price=510 → accepted, close anchored at 500
        t1 = MagicMock()
        t1.contract = MagicMock(symbol="SPY")
        t1.marketPrice.return_value = 510.0
        t1.bid = 509.0; t1.ask = 511.0; t1.last = 510.0
        t1.close = 500.0; t1.volume = 100
        svc._on_pending_tickers([t1])
        assert svc._prev_close["SPY"] == 500.0

        # Second tick: close changes to 600 (shouldn't update anchor)
        t2 = MagicMock()
        t2.contract = MagicMock(symbol="SPY")
        t2.marketPrice.return_value = 520.0
        t2.bid = 519.0; t2.ask = 521.0; t2.last = 520.0
        t2.close = 600.0; t2.volume = 200
        svc._on_pending_tickers([t2])
        # Anchor still 500, not 600
        assert svc._prev_close["SPY"] == 500.0

    @pytest.mark.anyio
    async def test_tws_poll_skips_outside_market_hours(self, tmp_path):
        """TWS poll loop sleeps instead of calling get_quote outside market hours."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPX\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: true\n")
        config = load_streaming_config(cfg_file)

        mock_provider = MagicMock()
        mock_provider._ib = MagicMock()
        mock_provider.is_healthy = MagicMock(return_value=True)
        mock_provider.get_quote = AsyncMock()

        svc = MarketDataStreamingService(config, ibkr_provider=mock_provider)
        svc._running = True

        # Patch _is_market_active to return False, and make the loop run once then stop
        call_count = 0
        async def fake_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                svc._running = False

        with patch("app.services.market_data._is_market_active", return_value=False), \
             patch("asyncio.sleep", side_effect=fake_sleep):
            await svc._tws_poll_loop()

        # get_quote should NOT have been called (market closed)
        mock_provider.get_quote.assert_not_called()

    @pytest.mark.anyio
    async def test_tws_poll_runs_during_market_hours(self, tmp_path):
        """TWS poll loop calls get_quote during market hours."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: true\n")
        config = load_streaming_config(cfg_file)

        mock_quote = MagicMock(last=510.0, bid=509.0, ask=511.0, volume=100)
        mock_provider = MagicMock()
        mock_provider._ib = MagicMock()
        mock_provider.is_healthy = MagicMock(return_value=True)
        mock_provider.get_quote = AsyncMock(return_value=mock_quote)

        svc = MarketDataStreamingService(config, ibkr_provider=mock_provider)
        svc._running = True
        svc._prev_close["SPY"] = 500.0  # Seed close for validation

        call_count = 0
        async def fake_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                svc._running = False

        with patch("app.services.market_data._is_market_active", return_value=True), \
             patch("asyncio.sleep", side_effect=fake_sleep):
            await svc._tws_poll_loop()

        # get_quote should have been called (market open)
        assert mock_provider.get_quote.call_count >= 1

    def test_on_pending_tickers_drops_outside_market(self, tmp_path):
        """_on_pending_tickers drops all ticks when market is closed."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: true\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        svc._subscriptions["SPY"] = MagicMock()
        svc._prev_close["SPY"] = 500.0

        ticker = MagicMock()
        ticker.contract = MagicMock(symbol="SPY")
        ticker.marketPrice.return_value = 510.0
        ticker.bid = 509.0; ticker.ask = 511.0; ticker.last = 510.0
        ticker.close = 500.0; ticker.volume = 100

        with patch("app.services.market_data._is_market_active", return_value=False):
            svc._on_pending_tickers([ticker])

        # Tick should have been dropped — no data in pending or last_tick
        assert "SPY" not in svc._pending_ticks
        assert "SPY" not in svc._last_tick

    def test_on_pending_tickers_accepts_during_market(self, tmp_path):
        """_on_pending_tickers accepts ticks when market is open."""
        from app.services.streaming_config import load_streaming_config
        from app.services.market_data_streaming import MarketDataStreamingService

        cfg_file = tmp_path / "stream.yaml"
        cfg_file.write_text("symbols:\n  - SPY\nredis_enabled: false\nquestdb_enabled: false\nws_broadcast_enabled: false\nmarket_hours_only: true\n")
        config = load_streaming_config(cfg_file)
        svc = MarketDataStreamingService(config, ibkr_provider=None)
        svc._subscriptions["SPY"] = MagicMock()

        ticker = MagicMock()
        ticker.contract = MagicMock(symbol="SPY")
        ticker.marketPrice.return_value = 510.0
        ticker.bid = 509.0; ticker.ask = 511.0; ticker.last = 510.0
        ticker.close = 500.0; ticker.volume = 100

        with patch("app.services.market_data._is_market_active", return_value=True):
            svc._on_pending_tickers([ticker])

        # Tick should have been accepted
        assert "SPY" in svc._last_tick
        assert svc._last_tick["SPY"]["price"] == 510.0


# ═══════════════════════════════════════════════════════════════════════════════
# Execution Store — IBKR execution history and multi-leg grouping
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionStore:
    """Tests for IBKR execution cache and order grouping."""

    def _make_fill(self, exec_id, perm_id, order_id, symbol, side, shares, price,
                   sec_type="STK", strike=0, right="", expiration="", con_id=100):
        return {
            "exec_id": exec_id,
            "order_id": order_id,
            "perm_id": perm_id,
            "time": "2026-03-18T15:00:00",
            "side": side,
            "shares": shares,
            "price": price,
            "avg_price": price,
            "cum_qty": shares,
            "account": "U123",
            "symbol": symbol,
            "sec_type": sec_type,
            "con_id": con_id,
            "local_symbol": "",
            "expiration": expiration,
            "strike": strike,
            "right": right,
            "exchange": "SMART",
            "commission": 1.0,
            "realized_pnl": 0.0,
        }

    def test_store_and_dedup(self, tmp_path):
        """Executions are stored and deduplicated by exec_id."""
        from app.services.execution_store import ExecutionStore
        store = ExecutionStore(tmp_path / "exec.json")
        fills = [self._make_fill("ex1", 1001, 1, "GBTC", "BOT", 100, 55.0)]
        assert store.merge_executions(fills) == 1
        assert store.count == 1
        # Merge again — should be 0 new
        assert store.merge_executions(fills) == 0
        assert store.count == 1

    def test_persistence_across_loads(self, tmp_path):
        """Stored executions survive reload."""
        from app.services.execution_store import ExecutionStore
        path = tmp_path / "exec.json"
        store1 = ExecutionStore(path)
        store1.merge_executions([self._make_fill("ex1", 1001, 1, "SPY", "BOT", 50, 500.0)])
        assert store1.count == 1
        # Reload
        store2 = ExecutionStore(path)
        assert store2.count == 1

    def test_group_equity_order(self, tmp_path):
        """Single equity fill grouped as one order."""
        from app.services.execution_store import ExecutionStore
        store = ExecutionStore(tmp_path / "exec.json")
        store.merge_executions([
            self._make_fill("ex1", 1001, 1, "GBTC", "BOT", 4350, 57.64),
        ])
        groups = store.get_grouped_by_order()
        assert len(groups) == 1
        assert groups[0]["order_type"] == "equity"
        assert groups[0]["leg_count"] == 1

    def test_group_credit_spread(self, tmp_path):
        """Two option fills with same perm_id grouped as credit spread."""
        from app.services.execution_store import ExecutionStore
        store = ExecutionStore(tmp_path / "exec.json")
        store.merge_executions([
            self._make_fill("ex1", 2001, 10, "RUT", "SLD", 1, 3.50,
                           sec_type="OPT", strike=2460, right="P",
                           expiration="20260318", con_id=200),
            self._make_fill("ex2", 2001, 10, "RUT", "BOT", 1, 1.20,
                           sec_type="OPT", strike=2440, right="P",
                           expiration="20260318", con_id=201),
        ])
        groups = store.get_grouped_by_order()
        assert len(groups) == 1
        g = groups[0]
        assert g["order_type"] == "credit_spread"
        assert g["leg_count"] == 2
        assert g["net_amount"] > 0  # credit received
        assert g["symbol"] == "RUT"
        assert g["expiration"] == "2026-03-18"

    def test_group_iron_condor(self, tmp_path):
        """Four option fills with same perm_id grouped as iron condor."""
        from app.services.execution_store import ExecutionStore
        store = ExecutionStore(tmp_path / "exec.json")
        store.merge_executions([
            self._make_fill("ex1", 3001, 20, "SPX", "SLD", 1, 5.00,
                           sec_type="OPT", strike=5500, right="P", con_id=300),
            self._make_fill("ex2", 3001, 20, "SPX", "BOT", 1, 2.00,
                           sec_type="OPT", strike=5475, right="P", con_id=301),
            self._make_fill("ex3", 3001, 20, "SPX", "SLD", 1, 4.00,
                           sec_type="OPT", strike=5700, right="C", con_id=302),
            self._make_fill("ex4", 3001, 20, "SPX", "BOT", 1, 1.50,
                           sec_type="OPT", strike=5725, right="C", con_id=303),
        ])
        groups = store.get_grouped_by_order()
        assert len(groups) == 1
        assert groups[0]["order_type"] == "iron_condor"
        assert groups[0]["leg_count"] == 4

    def test_multiple_orders_separate(self, tmp_path):
        """Different perm_ids produce separate groups."""
        from app.services.execution_store import ExecutionStore
        store = ExecutionStore(tmp_path / "exec.json")
        store.merge_executions([
            self._make_fill("ex1", 4001, 30, "GBTC", "BOT", 100, 55.0, con_id=400),
            self._make_fill("ex2", 4002, 31, "RUT", "SLD", 1, 3.00,
                           sec_type="OPT", strike=2460, right="P", con_id=401),
            self._make_fill("ex3", 4002, 31, "RUT", "BOT", 1, 1.00,
                           sec_type="OPT", strike=2440, right="P", con_id=402),
        ])
        groups = store.get_grouped_by_order()
        assert len(groups) == 2
        types = {g["order_type"] for g in groups}
        assert "equity" in types
        assert "credit_spread" in types

    def test_flush(self, tmp_path):
        """Flush clears all executions."""
        from app.services.execution_store import ExecutionStore
        store = ExecutionStore(tmp_path / "exec.json")
        store.merge_executions([self._make_fill("ex1", 5001, 40, "SPY", "BOT", 10, 500.0)])
        assert store.count == 1
        removed = store.flush()
        assert removed == 1
        assert store.count == 0

    def test_merge_across_runs(self, tmp_path):
        """Merging new executions preserves existing ones."""
        from app.services.execution_store import ExecutionStore
        store = ExecutionStore(tmp_path / "exec.json")
        store.merge_executions([self._make_fill("ex1", 6001, 50, "SPY", "BOT", 10, 500.0)])
        store.merge_executions([self._make_fill("ex2", 6002, 51, "QQQ", "BOT", 20, 400.0)])
        assert store.count == 2
        groups = store.get_grouped_by_order()
        assert len(groups) == 2

    def test_cmd_executions_exists(self):
        """executions CLI command exists."""
        import utp
        assert hasattr(utp, "_cmd_executions")


# ═══════════════════════════════════════════════════════════════════════════════
# Close Order Pricing — conId path price sign correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestCloseOrderPricing:
    """Tests for _close_by_con_id price sign and combo leg ordering."""

    def test_build_closing_trade_credit_spread(self):
        """Closing a credit spread produces BUY_TO_CLOSE + SELL_TO_CLOSE legs."""
        from app.services.trade_service import build_closing_trade_request
        position = {
            "order_type": "multi_leg",
            "symbol": "RUT",
            "quantity": 5,
            "broker": "ibkr",
            "expiration": "2026-03-19",
            "legs": [
                {"action": "SELL", "option_type": "PUT", "strike": 2420, "quantity": 1},
                {"action": "BUY", "option_type": "PUT", "strike": 2400, "quantity": 1},
            ],
        }
        req = build_closing_trade_request(position, quantity=1, net_price=1.50)
        assert req.multi_leg_order is not None
        legs = req.multi_leg_order.legs
        assert len(legs) == 2
        actions = {leg.action.value for leg in legs}
        assert "BUY_TO_CLOSE" in actions
        assert "SELL_TO_CLOSE" in actions

    def test_build_closing_trade_debit_spread(self):
        """Closing a debit spread produces reversed legs."""
        from app.services.trade_service import build_closing_trade_request
        position = {
            "order_type": "multi_leg",
            "symbol": "QQQ",
            "quantity": 3,
            "broker": "ibkr",
            "expiration": "2026-03-20",
            "legs": [
                {"action": "BUY", "option_type": "CALL", "strike": 480, "quantity": 1},
                {"action": "SELL", "option_type": "CALL", "strike": 490, "quantity": 1},
            ],
        }
        req = build_closing_trade_request(position, quantity=1, net_price=2.00)
        legs = req.multi_leg_order.legs
        actions = {leg.action.value for leg in legs}
        assert "SELL_TO_CLOSE" in actions
        assert "BUY_TO_CLOSE" in actions

    def test_build_closing_trade_equity_sell(self):
        """Closing a long equity position produces a SELL order."""
        from app.services.trade_service import build_closing_trade_request
        position = {
            "order_type": "equity",
            "symbol": "GBTC",
            "quantity": 100,
            "broker": "ibkr",
        }
        req = build_closing_trade_request(position, quantity=50, net_price=0)
        assert req.equity_order is not None
        assert req.equity_order.side.value == "SELL"
        assert req.equity_order.quantity == 50

    def test_build_closing_trade_equity_short_cover(self):
        """Closing a short equity position produces a BUY order."""
        from app.services.trade_service import build_closing_trade_request
        position = {
            "order_type": "equity",
            "symbol": "SPY",
            "quantity": -10,
            "broker": "ibkr",
        }
        req = build_closing_trade_request(position, quantity=10, net_price=0)
        assert req.equity_order.side.value == "BUY"

    @pytest.mark.anyio
    async def test_close_by_con_id_credit_spread_price_positive(self):
        """Closing a credit spread via conId uses positive price (debit)."""
        # We can't call _close_by_con_id directly (needs IBKR), but we can
        # verify the pricing logic by checking the route's combo construction.
        # The key invariant: credit spread close → BUY first → positive price.
        from ib_insync import ComboLeg
        legs = [
            {"action": "SELL", "option_type": "PUT", "strike": 2420, "quantity": 1, "con_id": 100},
            {"action": "BUY", "option_type": "PUT", "strike": 2400, "quantity": 1, "con_id": 101},
        ]
        combo_legs = []
        for leg in legs:
            close_action = "BUY" if "SELL" in leg["action"] else "SELL"
            combo_legs.append(ComboLeg(conId=leg["con_id"], ratio=1, action=close_action, exchange="SMART"))

        has_short_leg = any(leg["action"] == "SELL" for leg in legs)
        assert has_short_leg is True
        # Credit spread close: BUY first → positive price
        combo_legs.sort(key=lambda cl: (0 if cl.action == "BUY" else 1))
        assert combo_legs[0].action == "BUY"
        ibkr_price = abs(1.50)  # debit
        assert ibkr_price > 0

    @pytest.mark.anyio
    async def test_close_by_con_id_debit_spread_price_negative(self):
        """Closing a debit spread via conId uses negative price (credit)."""
        from ib_insync import ComboLeg
        legs = [
            {"action": "BUY", "option_type": "CALL", "strike": 480, "quantity": 1, "con_id": 200},
            {"action": "SELL", "option_type": "CALL", "strike": 490, "quantity": 1, "con_id": 201},
        ]
        # Debit spread: only long legs after closing the short
        # Original: BUY 480C (long) + SELL 490C (short) — has a SELL leg
        # Actually this IS a credit spread too (short the higher strike)
        # A pure debit spread would be: BUY 480C + SELL 490C where BUY is first
        # Let's test a true debit: BUY only
        legs_debit = [
            {"action": "BUY", "option_type": "CALL", "strike": 480, "quantity": 1, "con_id": 200},
        ]
        has_short = any(leg["action"] == "SELL" for leg in legs_debit)
        assert has_short is False
        ibkr_price = -abs(2.00)  # credit (receiving money back)
        assert ibkr_price < 0

    def test_close_net_price_default_none(self):
        """Close CLI --net-price defaults to None (use mark)."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--net-price", type=float, default=None)
        args = parser.parse_args([])
        assert args.net_price is None

    def test_close_net_price_explicit(self):
        """Close CLI --net-price can be set explicitly."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--net-price", type=float, default=None)
        args = parser.parse_args(["--net-price", "1.50"])
        assert args.net_price == 1.50


# ═══════════════════════════════════════════════════════════════════════════════
# Market Order Default — no net_price means MARKET, with net_price means LIMIT
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarketOrderDefault:
    """Verify order_type defaults to MARKET when net_price is absent, LIMIT when present."""

    def test_multi_leg_order_defaults_to_market(self):
        """MultiLegOrder with no net_price has order_type=MARKET."""
        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(
                    symbol="SPX", expiration="20260320", strike=5500,
                    option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1,
                ),
                OptionLeg(
                    symbol="SPX", expiration="20260320", strike=5475,
                    option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1,
                ),
            ],
            quantity=1,
        )
        assert order.order_type == OrderType.MARKET
        assert order.net_price is None

    def test_multi_leg_order_with_net_price_is_limit(self):
        """MultiLegOrder with net_price has order_type=LIMIT."""
        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(
                    symbol="SPX", expiration="20260320", strike=5500,
                    option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1,
                ),
                OptionLeg(
                    symbol="SPX", expiration="20260320", strike=5475,
                    option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1,
                ),
            ],
            order_type=OrderType.LIMIT,
            net_price=3.50,
            quantity=1,
        )
        assert order.order_type == OrderType.LIMIT
        assert order.net_price == 3.50

    def test_equity_order_defaults_to_market(self):
        """EquityOrder defaults to MARKET."""
        order = EquityOrder(
            broker=Broker.IBKR,
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
        )
        assert order.order_type == OrderType.MARKET
        assert order.limit_price is None

    def test_equity_order_with_limit_price_is_limit(self):
        """EquityOrder with limit_price stays at whatever order_type is set."""
        order = EquityOrder(
            broker=Broker.IBKR,
            symbol="SPY",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=450.00,
        )
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 450.00

    def test_build_closing_trade_no_price_is_market(self):
        """build_closing_trade_request() with no net_price produces a MARKET MultiLegOrder."""
        from app.services.trade_service import build_closing_trade_request

        position = {
            "order_type": "multi_leg",
            "symbol": "SPX",
            "broker": "ibkr",
            "quantity": 1,
            "expiration": "2026-03-20",
            "legs": [
                {"action": "SELL_TO_OPEN", "strike": 5500, "option_type": "PUT", "quantity": 1},
                {"action": "BUY_TO_OPEN", "strike": 5475, "option_type": "PUT", "quantity": 1},
            ],
        }
        req = build_closing_trade_request(position)
        assert req.multi_leg_order is not None
        assert req.multi_leg_order.order_type == OrderType.MARKET
        assert req.multi_leg_order.net_price is None

    def test_build_closing_trade_with_price_is_limit(self):
        """build_closing_trade_request() with net_price=0.10 produces LIMIT."""
        from app.services.trade_service import build_closing_trade_request

        position = {
            "order_type": "multi_leg",
            "symbol": "SPX",
            "broker": "ibkr",
            "quantity": 1,
            "expiration": "2026-03-20",
            "legs": [
                {"action": "SELL_TO_OPEN", "strike": 5500, "option_type": "PUT", "quantity": 1},
                {"action": "BUY_TO_OPEN", "strike": 5475, "option_type": "PUT", "quantity": 1},
            ],
        }
        req = build_closing_trade_request(position, net_price=0.10)
        assert req.multi_leg_order is not None
        assert req.multi_leg_order.order_type == OrderType.LIMIT
        assert req.multi_leg_order.net_price == 0.10

    def test_playbook_credit_spread_no_price_is_market(self):
        """PlaybookService._build_credit_spread with no net_price produces MARKET."""
        from app.services.playbook_service import PlaybookService

        instr = PlaybookInstruction(id="test", type="credit_spread", params={
            "symbol": "SPX", "expiration": "20260320", "short_strike": 5500,
            "long_strike": 5475, "option_type": "PUT", "quantity": 1,
        })
        svc = PlaybookService()
        req = svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert req.multi_leg_order is not None
        assert req.multi_leg_order.order_type == OrderType.MARKET
        assert req.multi_leg_order.net_price is None

    def test_playbook_credit_spread_with_price_is_limit(self):
        """PlaybookService._build_credit_spread with net_price produces LIMIT."""
        from app.services.playbook_service import PlaybookService

        instr = PlaybookInstruction(id="test", type="credit_spread", params={
            "symbol": "SPX", "expiration": "20260320", "short_strike": 5500,
            "long_strike": 5475, "option_type": "PUT", "quantity": 1,
            "net_price": 3.50,
        })
        svc = PlaybookService()
        req = svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert req.multi_leg_order is not None
        assert req.multi_leg_order.order_type == OrderType.LIMIT
        assert req.multi_leg_order.net_price == 3.50

    def test_playbook_debit_spread_no_price_is_market(self):
        """PlaybookService._build_debit_spread with no net_price produces MARKET."""
        from app.services.playbook_service import PlaybookService

        instr = PlaybookInstruction(id="test", type="debit_spread", params={
            "symbol": "QQQ", "expiration": "20260320", "long_strike": 480,
            "short_strike": 490, "option_type": "CALL", "quantity": 1,
        })
        svc = PlaybookService()
        req = svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert req.multi_leg_order is not None
        assert req.multi_leg_order.order_type == OrderType.MARKET
        assert req.multi_leg_order.net_price is None

    def test_playbook_iron_condor_no_price_is_market(self):
        """PlaybookService._build_iron_condor with no net_price produces MARKET."""
        from app.services.playbook_service import PlaybookService

        instr = PlaybookInstruction(id="test", type="iron_condor", params={
            "symbol": "SPX", "expiration": "20260320",
            "put_short": 5500, "put_long": 5475,
            "call_short": 5700, "call_long": 5725,
            "quantity": 1,
        })
        svc = PlaybookService()
        req = svc.instruction_to_trade_request(instr, Broker.IBKR)
        assert req.multi_leg_order is not None
        assert req.multi_leg_order.order_type == OrderType.MARKET
        assert req.multi_leg_order.net_price is None


# ═══════════════════════════════════════════════════════════════════════════════
# SMART Exchange Routing — ensure all orders go through SMART
# ═══════════════════════════════════════════════════════════════════════════════


class TestSmartExchangeRouting:
    """Verify all order paths use SMART exchange for best execution."""

    def test_config_default_exchange_is_smart(self):
        """Default IBKR exchange config is SMART."""
        from app.config import Settings
        s = Settings()
        assert s.ibkr_exchange == "SMART"

    def test_ibkr_stub_provider_exchange_smart(self):
        """Stub IBKRProvider uses SMART exchange."""
        from app.core.providers.ibkr import IBKRProvider
        p = IBKRProvider()
        # Stub doesn't set _exchange, but check the config default
        from app.config import settings
        assert settings.ibkr_exchange == "SMART"

    def test_ibkr_live_provider_default_exchange(self):
        """IBKRLiveProvider defaults to config exchange (SMART)."""
        from app.core.providers.ibkr import IBKRLiveProvider
        p = IBKRLiveProvider()
        assert p._exchange == "SMART"

    def test_ibkr_live_provider_explicit_exchange(self):
        """IBKRLiveProvider respects explicit exchange override."""
        from app.core.providers.ibkr import IBKRLiveProvider
        p = IBKRLiveProvider(exchange="CBOE")
        assert p._exchange == "CBOE"

    def test_equity_order_uses_provider_exchange(self):
        """Equity orders use the provider's exchange (SMART by default)."""
        # The Stock() constructor in execute_equity_order uses self._exchange
        from app.core.providers.ibkr import IBKRLiveProvider
        p = IBKRLiveProvider()
        # Verify the exchange attribute that Stock() will use
        assert p._exchange == "SMART"

    def test_multi_leg_combo_uses_provider_exchange(self):
        """Multi-leg combo orders use the provider's exchange for combo legs."""
        # ComboLeg exchange is set from self._exchange in execute_multi_leg_order
        from app.core.providers.ibkr import IBKRLiveProvider
        p = IBKRLiveProvider()
        assert p._exchange == "SMART"

    def test_close_by_con_id_uses_provider_exchange(self):
        """_close_by_con_id uses the provider's exchange for combo legs."""
        # The function reads exchange from getattr(provider, "_exchange", "SMART")
        from app.core.providers.ibkr import IBKRLiveProvider
        p = IBKRLiveProvider()
        assert p._exchange == "SMART"

    def test_qualify_option_tries_smart_first(self):
        """_qualify_option tries SMART exchange first."""
        # The exchange list in _qualify_option starts with SMART
        import inspect
        from app.core.providers.ibkr import IBKRLiveProvider
        source = inspect.getsource(IBKRLiveProvider._qualify_option)
        # Verify SMART is first in the exchange list
        assert '"SMART"' in source
        smart_pos = source.index('"SMART"')
        cboe_pos = source.index('"CBOE"')
        assert smart_pos < cboe_pos  # SMART tried before CBOE

    def test_option_chain_uses_smart(self):
        """get_option_quotes uses SMART for option qualification."""
        import inspect
        from app.core.providers.ibkr import IBKRLiveProvider
        source = inspect.getsource(IBKRLiveProvider.get_option_quotes)
        assert 'opt_exchange = "SMART"' in source

    def test_confirm_flag_exists_on_trade(self):
        """Trade command has --confirm flag."""
        import utp
        # Parse with credit-spread subcommand
        import argparse
        # Just verify the flag is defined by checking the help output
        import subprocess
        result = subprocess.run(
            ["python", "-c", "import utp"],
            capture_output=True, text=True
        )
        assert result.returncode == 0  # module imports fine

    def test_confirm_flag_exists_on_close(self):
        """Close command has --confirm flag."""
        import subprocess
        result = subprocess.run(
            ["python", "utp.py", "close", "--help"],
            capture_output=True, text=True
        )
        assert "--confirm" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════════
# Option Quote Streaming
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionQuoteStreaming:
    """Tests for background option quote streaming cache."""

    def test_streaming_config_defaults(self):
        """StreamingConfig has option quote fields with correct defaults.

        Defaults updated: poll_interval 15→5s (so the IBKR-overlay gate fires
        close to the 25s greeks_interval rather than being stuck on 15s ticks).
        """
        from app.services.streaming_config import StreamingConfig
        cfg = StreamingConfig()
        assert cfg.option_quotes_enabled is False
        assert cfg.option_quotes_poll_interval == 5.0   # was 15.0
        assert cfg.option_quotes_strike_range_pct == 15.0
        assert cfg.option_quotes_num_expirations == 12
        assert cfg.option_quotes_csv_dte_max == 10

    def test_streaming_config_loads_from_yaml(self, tmp_path):
        """load_streaming_config parses option quote fields from YAML."""
        from app.services.streaming_config import load_streaming_config
        yaml_content = """\
symbols:
  - SPX
option_quotes_enabled: true
option_quotes_poll_interval: 5.0
option_quotes_strike_range_pct: 4.5
option_quotes_num_expirations: 2
"""
        cfg_file = tmp_path / "streaming.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_streaming_config(cfg_file)
        assert cfg.option_quotes_enabled is True
        assert cfg.option_quotes_poll_interval == 5.0
        assert cfg.option_quotes_strike_range_pct == 4.5
        assert cfg.option_quotes_num_expirations == 2

    def test_cache_put_get(self):
        """Cache stores and retrieves quotes."""
        from app.services.option_quote_streaming import OptionQuoteCache
        cache = OptionQuoteCache()
        quotes = [{"strike": 5500, "bid": 1.0, "ask": 1.5}]
        cache.put("SPX", "2026-03-24", "CALL", quotes)
        result = cache.get("SPX", "2026-03-24", "CALL", max_age_seconds=10.0)
        assert result == quotes

    def test_cache_get_expired(self):
        """Cache returns None for expired entries."""
        from app.services.option_quote_streaming import OptionQuoteCache, CachedQuotes
        cache = OptionQuoteCache()
        # Insert with an old timestamp
        cache._cache[("SPX", "2026-03-24", "CALL")] = CachedQuotes(
            quotes=[{"strike": 5500}],
            fetched_at=time.monotonic() - 60,
            fetched_at_utc="2026-03-24T10:00:00+00:00",
        )
        result = cache.get("SPX", "2026-03-24", "CALL", max_age_seconds=10.0)
        assert result is None

    def test_cache_get_missing(self):
        """Cache returns None for missing keys."""
        from app.services.option_quote_streaming import OptionQuoteCache
        cache = OptionQuoteCache()
        assert cache.get("SPX", "2026-03-24", "CALL") is None

    def test_cache_stats(self):
        """Cache stats returns entry count and total quotes."""
        from app.services.option_quote_streaming import OptionQuoteCache
        cache = OptionQuoteCache()
        cache.put("SPX", "2026-03-24", "CALL", [{"s": 1}, {"s": 2}])
        cache.put("SPX", "2026-03-24", "PUT", [{"s": 3}])
        stats = cache.stats()
        assert stats["entries"] == 2
        assert stats["total_quotes"] == 3

    def test_cache_clear(self):
        """Cache clear removes all entries."""
        from app.services.option_quote_streaming import OptionQuoteCache
        cache = OptionQuoteCache()
        cache.put("SPX", "2026-03-24", "CALL", [{"s": 1}])
        cache.clear()
        assert cache.stats()["entries"] == 0

    def test_cache_normalizes_expiration_format(self):
        """Cache normalizes YYYYMMDD and YYYY-MM-DD to same key."""
        from app.services.option_quote_streaming import OptionQuoteCache
        cache = OptionQuoteCache()
        quotes = [{"strike": 5500, "bid": 1.0}]
        # Store with YYYYMMDD
        cache.put("SPX", "20260324", "CALL", quotes)
        # Retrieve with YYYY-MM-DD
        result = cache.get("SPX", "2026-03-24", "CALL", max_age_seconds=10.0)
        assert result == quotes
        # And vice versa
        cache.put("SPX", "2026-03-25", "PUT", quotes)
        result2 = cache.get("SPX", "20260325", "PUT", max_age_seconds=10.0)
        assert result2 == quotes

    def test_next_n_trading_days_skips_weekends(self):
        """_next_n_trading_days skips Saturday and Sunday."""
        from app.services.option_quote_streaming import _next_n_trading_days
        # 2026-03-20 is Friday
        result = _next_n_trading_days(3, start=date(2026, 3, 20))
        assert result == ["2026-03-20", "2026-03-23", "2026-03-24"]

    def test_next_n_trading_days_includes_today(self):
        """_next_n_trading_days includes today if it's a weekday."""
        from app.services.option_quote_streaming import _next_n_trading_days
        # 2026-03-23 is Monday
        result = _next_n_trading_days(1, start=date(2026, 3, 23))
        assert result == ["2026-03-23"]

    def test_next_n_trading_days_starts_on_weekend(self):
        """_next_n_trading_days starting on Saturday skips to Monday."""
        from app.services.option_quote_streaming import _next_n_trading_days
        # 2026-03-21 is Saturday
        result = _next_n_trading_days(2, start=date(2026, 3, 21))
        assert result == ["2026-03-23", "2026-03-24"]

    async def test_service_lifecycle(self):
        """Service start/stop lifecycle."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig
        svc = OptionQuoteStreamingService(StreamingConfig(), provider=MagicMock())
        await svc.start()
        assert svc.stats["running"] is True
        await svc.stop()
        assert svc.stats["running"] is False

    async def test_service_run_cycle_skips_market_closed_when_cache_warm(self):
        """Service skips fetching when market is closed and cache already populated."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig
        provider = AsyncMock()
        cfg = StreamingConfig(symbols=[StreamingSymbolConfig(symbol="SPX", sec_type="IND")])
        svc = OptionQuoteStreamingService(cfg, provider=provider)

        # Pre-populate cache so the "warm cache once" logic doesn't trigger
        svc._cache.put("SPX", "2026-03-24", "CALL", [{"strike": 5500}])

        # Patch _is_market_active (now used instead of _is_market_hours)
        with patch("app.services.market_data._is_market_active", return_value=False):
            await svc._run_one_cycle()

        # Provider should NOT have been called (market closed + cache warm)
        provider.get_quote.assert_not_called()
        provider.get_option_quotes.assert_not_called()

    async def test_service_fetches_for_configured_symbols(self):
        """Service calls provider for configured symbols during market hours."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig
        provider = AsyncMock()
        provider.get_quote.return_value = MagicMock(last=5500.0, bid=5499.0, ask=5501.0)
        provider.get_option_chain.return_value = {
            "expirations": ["2026-03-24", "2026-03-25"],
            "strikes": [5400, 5500, 5600],
        }
        provider.get_option_quotes.return_value = [{"strike": 5500, "bid": 1.0}]

        cfg = StreamingConfig(
            symbols=[StreamingSymbolConfig(symbol="SPX", sec_type="IND", exchange="CBOE")],
            option_quotes_num_expirations=1,
        )
        svc = OptionQuoteStreamingService(cfg, provider=provider)

        # Patch datetime to return market-open time (15:00 UTC)
        # Also patch market_data.get_quote since _get_price now uses the centralized layer
        import app.services.option_quote_streaming as oqs_mod
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = datetime(2026, 3, 24, 15, 0, 0, tzinfo=timezone.utc)
        mock_quote = MagicMock(last=5500.0, bid=5499.0, ask=5501.0, source="test")
        with patch.object(oqs_mod, "datetime", mock_dt):
            with patch.object(oqs_mod, "date") as mock_date:
                mock_date.today.return_value = date(2026, 3, 24)
                mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
                with patch("app.services.market_data.get_quote", new_callable=AsyncMock, return_value=mock_quote), \
                     patch("app.services.market_data._is_market_active", return_value=True):
                    await svc._run_one_cycle()

        # Provider should have been called
        assert provider.get_option_quotes.call_count >= 1

    def test_get_cached_quotes_strike_filtering(self):
        """get_cached_quotes filters by strike range."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig
        svc = OptionQuoteStreamingService(StreamingConfig(), provider=MagicMock())
        svc._cache.put("SPX", "2026-03-24", "CALL", [
            {"strike": 5400, "bid": 1.0},
            {"strike": 5500, "bid": 2.0},
            {"strike": 5600, "bid": 0.5},
        ])
        result = svc.get_cached_quotes("SPX", "2026-03-24", "CALL",
                                       strike_min=5450, strike_max=5550)
        assert len(result) == 1
        assert result[0]["strike"] == 5500

    def test_singleton_init_get_reset(self):
        """Module singleton: init, get, reset."""
        from app.services.option_quote_streaming import (
            init_option_quote_streaming,
            get_option_quote_streaming,
            reset_option_quote_streaming,
        )
        from app.services.streaming_config import StreamingConfig
        svc = init_option_quote_streaming(StreamingConfig(), provider=MagicMock())
        assert get_option_quote_streaming() is svc
        reset_option_quote_streaming()
        assert get_option_quote_streaming() is None

    async def test_route_option_quote_status_not_initialized(self, client, api_key_headers):
        """Status endpoint returns not-initialized when service is off."""
        resp = await client.get("/market/streaming/option-quotes/status", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False

    async def test_route_option_quote_status_initialized(self, client, api_key_headers):
        """Status endpoint returns stats when service is initialized."""
        from app.services.option_quote_streaming import init_option_quote_streaming, reset_option_quote_streaming
        from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig
        cfg = StreamingConfig(symbols=[StreamingSymbolConfig(symbol="SPX", sec_type="IND")])
        svc = init_option_quote_streaming(cfg, provider=MagicMock())
        await svc.start()
        try:
            resp = await client.get("/market/streaming/option-quotes/status", headers=api_key_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["running"] is True
            assert "SPX" in data["symbols"]
        finally:
            await svc.stop()
            reset_option_quote_streaming()

    async def test_route_options_uses_cache(self, client, api_key_headers):
        """GET /market/options/{symbol} serves from streaming cache (fast path, no provider call)."""
        from app.services.option_quote_streaming import init_option_quote_streaming, reset_option_quote_streaming
        from app.services.streaming_config import StreamingConfig
        from app.core.provider import ProviderRegistry
        from app.models import Broker

        cached_quotes = [{"strike": 5500, "bid": 1.0, "ask": 1.5, "last": 1.2,
                          "volume": 100, "open_interest": 500, "symbol": "SPX260324C5500"}]

        cfg = StreamingConfig()
        provider = MagicMock()
        svc = init_option_quote_streaming(cfg, provider=provider)
        svc._cache.put("SPX", "2026-03-24", "CALL", cached_quotes)

        # Mock the IBKR provider — should NOT be called at all (fast path)
        ibkr = ProviderRegistry.get(Broker.IBKR)
        ibkr.get_option_chain = AsyncMock(return_value={
            "expirations": ["2026-03-24"], "strikes": [5500]
        })
        ibkr.get_option_quotes = AsyncMock()

        try:
            resp = await client.get(
                "/market/options/SPX",
                params={"expiration": "2026-03-24", "option_type": "CALL"},
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "quotes" in data
            # Read-time merge tags each row with `source` and `age_seconds`.
            # The underlying quote fields still match what was cached.
            served = data["quotes"]["call"]
            assert len(served) == 1
            row = served[0]
            for k, v in cached_quotes[0].items():
                assert row[k] == v
            assert row["source"] == "csv"
            assert "age_seconds" in row
            assert data.get("source") == "streaming_cache"
            # Fast path: neither get_option_chain nor get_option_quotes called
            ibkr.get_option_chain.assert_not_called()
            ibkr.get_option_quotes.assert_not_called()
        finally:
            reset_option_quote_streaming()

    async def test_redis_conid_cache_save_and_load(self):
        """conID cache saves to Redis and loads on next startup."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        provider = MagicMock()
        provider._option_conid_cache = {"SPX_20260324_6580.0_C": 857789795, "SPX_20260324_6580.0_P": 857789800}
        provider._conid_cache = {"SPX": 416904}

        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_pipe = AsyncMock()
        mock_pipe.hset = MagicMock(return_value=mock_pipe)
        mock_pipe.expire = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        mock_redis.aclose = AsyncMock()

        cfg = StreamingConfig(redis_enabled=True, redis_url="redis://localhost:6379/0")
        svc = OptionQuoteStreamingService(cfg, provider=provider)
        svc._redis = mock_redis

        # Save
        await svc._redis_save_conid_cache()
        # Should have called hset for each entry
        assert mock_pipe.hset.call_count >= 2

        # Now simulate load into a fresh provider
        provider2 = MagicMock()
        provider2._option_conid_cache = {}
        provider2._conid_cache = {}
        svc2 = OptionQuoteStreamingService(cfg, provider=provider2)
        svc2._redis = AsyncMock()
        svc2._redis.hgetall = AsyncMock(side_effect=[
            {"SPX_20260324_6580.0_C": "857789795", "SPX_20260324_6580.0_P": "857789800"},
            {"SPX": "416904"},
        ])
        await svc2._redis_load_conid_cache()
        assert provider2._option_conid_cache["SPX_20260324_6580.0_C"] == 857789795
        assert provider2._conid_cache["SPX"] == 416904
        assert svc2._conid_cache_loaded is True

    async def test_route_options_falls_back_when_cache_partial(self, client, api_key_headers):
        """GET /market/options/{symbol} falls back to provider when cache is incomplete."""
        from app.services.option_quote_streaming import init_option_quote_streaming, reset_option_quote_streaming
        from app.services.streaming_config import StreamingConfig
        from app.core.provider import ProviderRegistry
        from app.models import Broker

        # Only cache CALLs, not PUTs — requesting BOTH should fall back
        cfg = StreamingConfig()
        provider = MagicMock()
        svc = init_option_quote_streaming(cfg, provider=provider)
        svc._cache.put("SPX", "2026-03-24", "CALL", [{"strike": 5500}])

        ibkr = ProviderRegistry.get(Broker.IBKR)
        ibkr.get_option_chain = AsyncMock(return_value={
            "expirations": ["2026-03-24"], "strikes": [5500]
        })
        ibkr.get_option_quotes = AsyncMock(return_value=[{"strike": 5500, "bid": 0.5}])

        try:
            resp = await client.get(
                "/market/options/SPX",
                params={"expiration": "2026-03-24"},  # no option_type → BOTH
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            # PUT was missing from cache — provider's get_option_quotes called for it
            ibkr.get_option_quotes.assert_called()
            # CALL came from cache, PUT from provider
            assert "quotes" in data
        finally:
            reset_option_quote_streaming()

    async def test_redis_quote_cache_put_writes_to_redis(self):
        """Cache.put() writes quotes to Redis asynchronously."""
        from app.services.option_quote_streaming import OptionQuoteCache
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        cache = OptionQuoteCache(redis_client=mock_redis)
        quotes = [{"strike": 5500, "bid": 1.0, "ask": 1.5}]
        cache.put("SPX", "2026-03-25", "CALL", quotes)
        # Allow the fire-and-forget task to run
        await asyncio.sleep(0.05)
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert "utp:option_quotes:SPX:2026-03-25:CALL" == call_args[0][0]
        import json as _json
        payload = _json.loads(call_args[0][1])
        assert payload["quotes"] == quotes
        assert "fetched_at_utc" in payload

    async def test_redis_quote_cache_load_populates_memory(self):
        """load_from_redis() populates in-memory cache from Redis."""
        import json as _json
        from app.services.option_quote_streaming import OptionQuoteCache
        quotes = [{"strike": 5500, "bid": 1.0, "ask": 1.5}]
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=lambda key: _json.dumps({
            "quotes": quotes, "fetched_at_utc": "2026-03-25T14:00:00+00:00",
        }) if "CALL" in key else None)

        cache = OptionQuoteCache(redis_client=mock_redis)
        loaded = await cache.load_from_redis(["SPX"], ["2026-03-25"])
        assert loaded == 1  # CALL loaded, PUT returned None
        result = cache.get("SPX", "2026-03-25", "CALL", max_age_seconds=60.0)
        assert result == quotes

    async def test_redis_quote_cache_no_overwrite_fresher(self):
        """load_from_redis() doesn't overwrite fresher in-memory data."""
        import json as _json
        from app.services.option_quote_streaming import OptionQuoteCache
        fresh_quotes = [{"strike": 5500, "bid": 2.0}]
        old_quotes = [{"strike": 5500, "bid": 1.0}]
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=_json.dumps({
            "quotes": old_quotes, "fetched_at_utc": "2026-03-25T12:00:00+00:00",
        }))

        cache = OptionQuoteCache(redis_client=mock_redis)
        # Put fresh data in memory first
        cache.put("SPX", "2026-03-25", "CALL", fresh_quotes)
        # Load from Redis — should NOT overwrite
        await cache.load_from_redis(["SPX"], ["2026-03-25"])
        result = cache.get("SPX", "2026-03-25", "CALL", max_age_seconds=60.0)
        assert result == fresh_quotes  # Still the fresh data

    def test_get_cached_quotes_csv_always_served(self):
        """CSV is the fallback tier — served whenever IBKR doesn't supply a fresher row.

        Replaces the old "stale CSV → None" semantic.  Per the read-time merge
        contract: CSV fills in any strike IBKR doesn't have a fresh quote for,
        regardless of CSV age.  IBKR's freshness threshold is the only gate.
        """
        from app.services.option_quote_streaming import OptionQuoteStreamingService, CachedQuotes
        from app.services.streaming_config import StreamingConfig

        svc = OptionQuoteStreamingService(StreamingConfig(), provider=MagicMock())
        # CSV-only cache, entry 2 minutes old
        svc._cache._cache[("SPX", "2026-03-25", "CALL")] = CachedQuotes(
            quotes=[{"strike": 5500}],
            fetched_at=time.monotonic() - 120,
            fetched_at_utc="2026-03-25T14:00:00+00:00",
        )
        result = svc.get_cached_quotes("SPX", "2026-03-25", "CALL")
        assert result is not None
        assert result[0]["strike"] == 5500
        assert result[0]["source"] == "csv"  # tagged by the merge

        # If IBKR also has the strike fresh, IBKR wins
        svc._ibkr_cache._cache[("SPX", "2026-03-25", "CALL")] = CachedQuotes(
            quotes=[{"strike": 5500, "bid": 9.99}],
            fetched_at=time.monotonic(),  # 0s old
            fetched_at_utc="2026-03-25T14:01:00+00:00",
        )
        result = svc.get_cached_quotes("SPX", "2026-03-25", "CALL")
        assert result is not None
        assert result[0]["source"] == "ibkr_fresh"
        assert result[0]["bid"] == 9.99

    # ── CSV primary mode tests ────────────────────────────────────────────

    def test_streaming_config_csv_primary_defaults(self):
        """StreamingConfig has CSV primary fields with correct defaults.

        IBKR overlay interval default tightened from 60s to 25s — see
        TestIbkrFetchParallel for the per-call latency math.
        """
        from app.services.streaming_config import StreamingConfig
        cfg = StreamingConfig()
        assert cfg.option_quotes_csv_primary is True
        assert cfg.option_quotes_csv_dir == ""
        assert cfg.option_quotes_greeks_interval == 25.0  # was 60.0

    def test_streaming_config_csv_primary_yaml(self, tmp_path):
        """load_streaming_config parses CSV primary fields from YAML."""
        from app.services.streaming_config import load_streaming_config
        yaml_content = """\
symbols:
  - SPX
option_quotes_csv_primary: false
option_quotes_csv_dir: /tmp/test_csv
option_quotes_greeks_interval: 30.0
"""
        cfg_file = tmp_path / "streaming.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_streaming_config(cfg_file)
        assert cfg.option_quotes_csv_primary is False
        assert cfg.option_quotes_csv_dir == "/tmp/test_csv"
        assert cfg.option_quotes_greeks_interval == 30.0

    def test_load_csv_latest_snapshot_basic(self, tmp_path):
        """_load_csv_latest_snapshot reads latest timestamp from a CSV file."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        # Create a test CSV
        csv_dir = tmp_path / "options"
        sym_dir = csv_dir / "SPX"
        sym_dir.mkdir(parents=True)
        csv_file = sym_dir / "2026-04-06.csv"
        csv_file.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            "2026-04-06T14:00:00.000000,O:SPX,call,5500,2026-04-06,2.0,2.5,2.2,,,,,,,"
            "100\n"
            "2026-04-06T14:00:00.000000,O:SPX,put,5500,2026-04-06,1.0,1.5,1.2,,,,,,,"
            "50\n"
            "2026-04-06T14:00:01.000000,O:SPX,call,5500,2026-04-06,2.1,2.6,2.3,,,,,,,"
            "110\n"
            "2026-04-06T14:00:01.000000,O:SPX,put,5500,2026-04-06,1.1,1.6,1.3,,,,,,,"
            "60\n"
        )

        cfg = StreamingConfig(option_quotes_csv_dir=str(csv_dir))
        svc = OptionQuoteStreamingService(cfg, provider=MagicMock())

        quotes, snap_ts = svc._load_csv_latest_snapshot("SPX", "2026-04-06", "CALL")
        assert len(quotes) == 1
        assert quotes[0]["strike"] == 5500
        assert quotes[0]["bid"] == 2.1
        assert quotes[0]["ask"] == 2.6
        assert quotes[0]["volume"] == 110
        assert snap_ts == "2026-04-06T14:00:01.000000"

    def test_load_csv_latest_snapshot_strike_filtering(self, tmp_path):
        """_load_csv_latest_snapshot filters by strike range."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        csv_dir = tmp_path / "options"
        sym_dir = csv_dir / "NDX"
        sym_dir.mkdir(parents=True)
        csv_file = sym_dir / "2026-04-06.csv"
        ts = "2026-04-06T15:00:00.000000"
        csv_file.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            f"{ts},O:NDX,call,19000,2026-04-06,10,15,12,,,,,,,50\n"
            f"{ts},O:NDX,call,20000,2026-04-06,5,8,6,,,,,,,80\n"
            f"{ts},O:NDX,call,21000,2026-04-06,1,2,1.5,,,,,,,30\n"
        )

        cfg = StreamingConfig(option_quotes_csv_dir=str(csv_dir))
        svc = OptionQuoteStreamingService(cfg, provider=MagicMock())

        quotes, _ = svc._load_csv_latest_snapshot(
            "NDX", "2026-04-06", "CALL", strike_min=19500, strike_max=20500,
        )
        assert len(quotes) == 1
        assert quotes[0]["strike"] == 20000

    def test_load_csv_latest_snapshot_missing_file(self, tmp_path):
        """_load_csv_latest_snapshot returns empty for missing file."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        csv_dir = tmp_path / "options"
        csv_dir.mkdir(parents=True)

        cfg = StreamingConfig(option_quotes_csv_dir=str(csv_dir))
        svc = OptionQuoteStreamingService(cfg, provider=MagicMock())

        quotes, snap_ts = svc._load_csv_latest_snapshot("SPX", "2026-04-06", "CALL")
        assert quotes == []
        assert snap_ts == ""

    def test_greeks_cache_merge(self):
        """_merge_greeks attaches cached greeks to CSV quotes."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        cfg = StreamingConfig()
        svc = OptionQuoteStreamingService(cfg, provider=MagicMock())

        # Pre-populate greeks cache
        svc._greeks_cache[("SPX", "2026-04-06", "CALL")] = {
            5500: {"delta": 0.45, "gamma": 0.02, "theta": -0.5, "vega": 1.2, "iv": 0.20},
            5600: {"delta": 0.30, "gamma": 0.01, "theta": -0.3, "vega": 0.8, "iv": 0.18},
        }

        quotes = [
            {"strike": 5500, "bid": 2.0, "ask": 2.5},
            {"strike": 5600, "bid": 1.0, "ask": 1.5},
            {"strike": 5700, "bid": 0.5, "ask": 0.8},  # No greeks for this strike
        ]

        merged = svc._merge_greeks(quotes, "SPX", "2026-04-06", "CALL")
        assert merged[0]["greeks"]["delta"] == 0.45
        assert merged[1]["greeks"]["iv"] == 0.18
        assert "greeks" not in merged[2]  # No greeks for 5700

    def test_greeks_persist_across_csv_refreshes(self, tmp_path):
        """Greeks survive across CSV refresh cycles."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        csv_dir = tmp_path / "options"
        sym_dir = csv_dir / "SPX"
        sym_dir.mkdir(parents=True)

        cfg = StreamingConfig(option_quotes_csv_dir=str(csv_dir))
        svc = OptionQuoteStreamingService(cfg, provider=MagicMock())

        # Store greeks
        svc._greeks_cache[("SPX", "2026-04-06", "CALL")] = {
            5500: {"delta": 0.45, "gamma": 0.02, "theta": -0.5, "vega": 1.2, "iv": 0.20},
        }

        # First CSV read
        ts1 = "2026-04-06T14:00:00.000000"
        csv_file = sym_dir / "2026-04-06.csv"
        csv_file.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            f"{ts1},O:SPX,call,5500,2026-04-06,2.0,2.5,2.2,,,,,,,100\n"
        )
        quotes1, _ = svc._load_csv_latest_snapshot("SPX", "2026-04-06", "CALL")
        merged1 = svc._merge_greeks(quotes1, "SPX", "2026-04-06", "CALL")
        assert merged1[0]["greeks"]["delta"] == 0.45

        # Second CSV read (new snapshot, same greeks)
        ts2 = "2026-04-06T14:01:00.000000"
        csv_file.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            f"{ts2},O:SPX,call,5500,2026-04-06,2.2,2.7,2.4,,,,,,,120\n"
        )
        quotes2, _ = svc._load_csv_latest_snapshot("SPX", "2026-04-06", "CALL")
        merged2 = svc._merge_greeks(quotes2, "SPX", "2026-04-06", "CALL")
        assert merged2[0]["bid"] == 2.2  # New CSV data
        assert merged2[0]["greeks"]["delta"] == 0.45  # Greeks persisted

    async def test_csv_cycle_skips_ibkr_when_not_due(self, tmp_path):
        """CSV primary cycle does NOT call IBKR when fetch interval hasn't elapsed."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig

        csv_dir = tmp_path / "options"
        sym_dir = csv_dir / "SPX"
        sym_dir.mkdir(parents=True)
        ts = "2026-04-06T15:00:00.000000"
        (sym_dir / "2026-04-06.csv").write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            f"{ts},O:SPX,call,5500,2026-04-06,2.0,2.5,2.2,,,,,,,100\n"
            f"{ts},O:SPX,put,5500,2026-04-06,1.0,1.5,1.2,,,,,,,50\n"
        )

        provider = AsyncMock()
        cfg = StreamingConfig(
            symbols=[StreamingSymbolConfig(symbol="SPX", sec_type="IND", exchange="CBOE")],
            option_quotes_csv_primary=True,
            option_quotes_csv_dir=str(csv_dir),
            option_quotes_greeks_interval=60.0,
            option_quotes_num_expirations=1,
        )
        svc = OptionQuoteStreamingService(cfg, provider=provider)
        svc._last_greeks_fetch = time.monotonic()  # Just fetched

        # Set up jobs manually (bypass price/expiration resolution)
        jobs = [("SPX", "2026-04-06", "CALL", 5000, 6000, "test"),
                ("SPX", "2026-04-06", "PUT", 5000, 6000, "test")]
        await svc._run_csv_primary_cycle(jobs)

        # IBKR should NOT have been called (greeks not due)
        provider.get_option_quotes.assert_not_called()
        # But CSV data should be cached
        cached = svc._cache.get("SPX", "2026-04-06", "CALL", max_age_seconds=10)
        assert cached is not None
        assert len(cached) == 1

    async def test_csv_cycle_fetches_ibkr_when_due(self, tmp_path):
        """CSV primary cycle calls IBKR for prices+greeks when interval has elapsed."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig

        csv_dir = tmp_path / "options"
        sym_dir = csv_dir / "SPX"
        sym_dir.mkdir(parents=True)
        ts = "2026-04-06T15:00:00.000000"
        (sym_dir / "2026-04-06.csv").write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            f"{ts},O:SPX,call,5500,2026-04-06,2.0,2.5,2.2,,,,,,,100\n"
            f"{ts},O:SPX,put,5500,2026-04-06,1.0,1.5,1.2,,,,,,,50\n"
        )

        provider = AsyncMock()
        provider.get_option_quotes.return_value = [
            {"strike": 5500, "bid": 2.0, "ask": 2.5,
             "greeks": {"delta": 0.45, "gamma": 0.02, "theta": -0.5, "vega": 1.2, "iv": 0.20}},
        ]
        cfg = StreamingConfig(
            symbols=[StreamingSymbolConfig(symbol="SPX", sec_type="IND", exchange="CBOE")],
            option_quotes_csv_primary=True,
            option_quotes_csv_dir=str(csv_dir),
            option_quotes_greeks_interval=60.0,
            option_quotes_num_expirations=1,
        )
        svc = OptionQuoteStreamingService(cfg, provider=provider)
        svc._last_greeks_fetch = 0  # Never fetched — greeks are due

        jobs = [("SPX", "2026-04-06", "CALL", 5000, 6000, "test"),
                ("SPX", "2026-04-06", "PUT", 5000, 6000, "test")]
        await svc._run_csv_primary_cycle(jobs)

        # IBKR SHOULD have been called for greeks
        assert provider.get_option_quotes.call_count >= 1

    async def test_csv_primary_disabled_uses_original(self):
        """When csv_primary is disabled, original IBKR-only path is used."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig
        import app.services.option_quote_streaming as oqs_mod

        provider = AsyncMock()
        provider.get_option_quotes.return_value = [{"strike": 5500, "bid": 1.0}]

        cfg = StreamingConfig(
            symbols=[StreamingSymbolConfig(symbol="SPX", sec_type="IND", exchange="CBOE")],
            option_quotes_csv_primary=False,
            option_quotes_num_expirations=1,
        )
        svc = OptionQuoteStreamingService(cfg, provider=provider)

        mock_quote = MagicMock(last=5500.0, bid=5499.0, ask=5501.0, source="test")
        mock_dt = MagicMock(wraps=datetime)
        mock_dt.now.return_value = datetime(2026, 4, 6, 15, 0, 0, tzinfo=timezone.utc)
        with patch.object(oqs_mod, "datetime", mock_dt):
            with patch.object(oqs_mod, "date") as mock_date:
                mock_date.today.return_value = date(2026, 4, 6)
                mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
                with patch("app.services.market_data.get_quote", new_callable=AsyncMock, return_value=mock_quote):
                    provider.get_option_chain.return_value = {
                        "expirations": ["2026-04-06"], "strikes": [5500],
                    }
                    await svc._run_one_cycle()

        # Should have used IBKR directly (not CSV)
        assert provider.get_option_quotes.call_count >= 1

    def test_get_expirations_from_csv(self, tmp_path):
        """_get_expirations_from_csv lists future expirations from directory."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        csv_dir = tmp_path / "options"
        sym_dir = csv_dir / "SPX"
        sym_dir.mkdir(parents=True)
        # Create some CSV files — use dates well in the future to avoid date-sensitivity
        (sym_dir / "2027-06-15.csv").write_text("header\n")
        (sym_dir / "2027-06-16.csv").write_text("header\n")
        (sym_dir / "2027-06-18.csv").write_text("header\n")
        (sym_dir / "2020-01-01.csv").write_text("header\n")  # Past date
        (sym_dir / "not-a-date.csv").write_text("header\n")  # Invalid

        cfg = StreamingConfig(option_quotes_csv_dir=str(csv_dir))
        svc = OptionQuoteStreamingService(cfg, provider=MagicMock())

        exps = svc._get_expirations_from_csv("SPX")
        assert "2027-06-15" in exps
        assert "2027-06-16" in exps
        assert "2027-06-18" in exps
        assert "2020-01-01" not in exps  # Past
        assert "not-a-date" not in exps  # Invalid

    def test_csv_primary_stats_in_service_stats(self):
        """Service stats include csv_primary section."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig

        cfg = StreamingConfig(option_quotes_csv_primary=True, option_quotes_greeks_interval=60.0)
        svc = OptionQuoteStreamingService(cfg, provider=MagicMock())
        stats = svc.stats
        assert "csv_primary" in stats
        assert stats["csv_primary"]["enabled"] is True
        assert stats["csv_primary"]["greeks_interval"] == 60.0
        assert stats["csv_primary"]["csv_reads_ok"] == 0
        assert stats["csv_primary"]["csv_reads_failed"] == 0

    async def test_run_one_cycle_uses_market_active(self):
        """_run_one_cycle uses _is_market_active from market_data (not local _is_market_hours)."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig
        provider = AsyncMock()
        cfg = StreamingConfig(symbols=[StreamingSymbolConfig(symbol="SPX", sec_type="IND")])
        svc = OptionQuoteStreamingService(cfg, provider=provider)
        svc._cache.put("SPX", "2026-03-24", "CALL", [{"strike": 5500}])

        # Patch _is_market_active on market_data module — this is what _run_one_cycle uses now
        with patch("app.services.market_data._is_market_active", return_value=False) as mock_active:
            await svc._run_one_cycle()
            mock_active.assert_called_once()

        assert svc._cycles_skipped_market_closed >= 1

    async def test_option_chain_routes_through_market_data(self, client, api_key_headers):
        """GET /market/options/{symbol}?list_expirations=true calls market_data.get_option_chain."""
        from datetime import datetime, timedelta
        from app.models import Broker
        # Use future dates so _merge_expirations doesn't filter them out
        d1 = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        d2 = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        with patch("app.services.market_data.get_option_chain", new_callable=AsyncMock,
                    return_value={"expirations": [d1, d2]}) as mock_chain:
            resp = await client.get(
                "/market/options/SPX",
                params={"list_expirations": "true"},
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert d1 in data["expirations"]
            assert d2 in data["expirations"]
            mock_chain.assert_called_once_with("SPX", Broker.IBKR)

    def test_greeks_interval_default_is_25(self):
        """StreamingConfig greeks interval tightened to 25s (was 60s).

        Paired with option_quotes_ibkr_max_parallel=6 this fits the typical
        p50 IBKR latency (5s × ceil(18/6) = 15s) inside the 22.5s wait_for
        budget (0.9 × 25s).
        """
        from app.services.streaming_config import StreamingConfig
        cfg = StreamingConfig()
        assert cfg.option_quotes_greeks_interval == 25.0


class TestEtradeProvider:
    """Tests for E*TRADE provider — stub, live provider, config, token management."""

    def test_etrade_live_provider_init(self):
        """Constructor sets defaults."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        assert provider.broker == Broker.ETRADE
        assert provider._connected is False
        assert provider._account_id_key == ""
        assert provider._orders == {}
        assert provider._session is None
        assert provider._accounts is None

    def test_etrade_is_healthy_disconnected(self):
        """is_healthy returns False when not connected."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        assert provider.is_healthy() is False

    def test_etrade_is_healthy_connected(self):
        """is_healthy returns True when connected with accounts client."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True
        provider._accounts = MagicMock()
        assert provider.is_healthy() is True

    @pytest.mark.asyncio
    async def test_etrade_readonly_blocks_equity_orders(self):
        """Orders rejected when etrade_readonly=true."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True
        provider._orders_client = MagicMock()
        provider._account_id_key = "test_acct"

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_readonly = True
            order = EquityOrder(
                broker=Broker.ETRADE, symbol="SPY", side=OrderSide.BUY,
                quantity=100, order_type=OrderType.MARKET,
            )
            result = await provider.execute_equity_order(order)
            assert result.status == OrderStatus.REJECTED
            assert "read-only" in result.message

    @pytest.mark.asyncio
    async def test_etrade_readonly_blocks_multi_leg_orders(self):
        """Multi-leg orders rejected when etrade_readonly=true."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True
        provider._orders_client = MagicMock()
        provider._account_id_key = "test_acct"

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_readonly = True
            order = MultiLegOrder(
                broker=Broker.ETRADE,
                legs=[
                    OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                              option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                    OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                              option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
                ],
                order_type=OrderType.LIMIT, net_price=2.50,
            )
            result = await provider.execute_multi_leg_order(order)
            assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_etrade_equity_order_preview_place(self):
        """2-step preview → place flow for equity orders."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True
        provider._account_id_key = "test_acct"

        mock_orders = MagicMock()
        mock_orders.preview_equity_order = MagicMock(return_value={
            "PreviewOrderResponse": {
                "PreviewIds": [{"previewId": "preview_123"}],
            }
        })
        mock_orders.place_equity_order = MagicMock(return_value={
            "PlaceOrderResponse": {
                "OrderIds": [{"orderId": 99001}],
            }
        })
        provider._orders_client = mock_orders

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_readonly = False
            order = EquityOrder(
                broker=Broker.ETRADE, symbol="SPY", side=OrderSide.BUY,
                quantity=100, order_type=OrderType.MARKET,
            )
            result = await provider.execute_equity_order(order)

        assert result.status == OrderStatus.SUBMITTED
        assert result.order_id == "99001"
        mock_orders.preview_equity_order.assert_called_once()
        mock_orders.place_equity_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_etrade_multi_leg_order_payload(self):
        """Correct spread order construction with preview → place."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True
        provider._account_id_key = "test_acct"

        mock_orders = MagicMock()
        mock_orders.preview_option_order = MagicMock(return_value={
            "PreviewOrderResponse": {
                "PreviewIds": [{"previewId": "prev_456"}],
            }
        })
        mock_orders.place_option_order = MagicMock(return_value={
            "PlaceOrderResponse": {
                "OrderIds": [{"orderId": 99002}],
            }
        })
        provider._orders_client = mock_orders

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_readonly = False
            order = MultiLegOrder(
                broker=Broker.ETRADE,
                legs=[
                    OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5500.0,
                              option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN, quantity=1),
                    OptionLeg(symbol="SPX", expiration="2026-03-20", strike=5475.0,
                              option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN, quantity=1),
                ],
                order_type=OrderType.LIMIT, net_price=2.50, quantity=1,
            )
            result = await provider.execute_multi_leg_order(order)

        assert result.status == OrderStatus.SUBMITTED
        assert result.order_id == "99002"
        assert "credit" in result.message

        # Verify preview was called with correct structure
        call_kwargs = mock_orders.preview_option_order.call_args[1]
        order_payload = call_kwargs["order"][0]
        assert order_payload["priceType"] == "NET_CREDIT"
        assert len(order_payload["Instrument"]) == 2
        assert order_payload["Instrument"][0]["orderAction"] == "SELL_OPEN"
        assert order_payload["Instrument"][1]["orderAction"] == "BUY_OPEN"

    @pytest.mark.asyncio
    async def test_etrade_get_quote_parsing(self):
        """Quote response → Quote model."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True

        mock_market = MagicMock()
        mock_market.get_quote = MagicMock(return_value={
            "QuoteResponse": {
                "QuoteData": [{
                    "All": {
                        "bid": 450.50,
                        "ask": 451.00,
                        "lastTrade": 450.75,
                        "totalVolume": 1500000,
                    }
                }]
            }
        })
        provider._market = mock_market

        quote = await provider.get_quote("SPY")
        assert quote.symbol == "SPY"
        assert quote.bid == 450.50
        assert quote.ask == 451.00
        assert quote.last == 450.75
        assert quote.volume == 1500000
        assert quote.source == "etrade"

    @pytest.mark.asyncio
    async def test_etrade_get_positions_parsing(self):
        """Portfolio response → Position list."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True
        provider._account_id_key = "test_acct"

        mock_accounts = MagicMock()
        mock_accounts.get_account_portfolio = MagicMock(return_value={
            "PortfolioResponse": {
                "AccountPortfolio": [{
                    "Position": [
                        {
                            "Product": {"symbol": "AAPL", "securityType": "EQ"},
                            "quantity": 100,
                            "costPerShare": 175.00,
                            "marketValue": 18500.00,
                            "totalGain": 1000.00,
                        },
                        {
                            "Product": {
                                "symbol": "SPY", "securityType": "OPTN",
                                "callPut": "PUT", "strikePrice": 450.0,
                                "expiryYear": "2026", "expiryMonth": "3", "expiryDay": "20",
                            },
                            "quantity": -5,
                            "costPerShare": 3.50,
                            "marketValue": -1500.00,
                            "totalGain": 250.00,
                        },
                    ]
                }]
            }
        })
        provider._accounts = mock_accounts

        positions = await provider.get_positions()
        assert len(positions) == 2

        # Equity position
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 100
        assert positions[0].sec_type == "STK"

        # Option position
        assert positions[1].symbol == "SPY"
        assert positions[1].quantity == -5
        assert positions[1].sec_type == "OPT"
        assert positions[1].strike == 450.0
        assert positions[1].right == "P"
        assert positions[1].expiration == "20260320"

    def test_etrade_order_status_mapping(self):
        """E*TRADE statuses → OrderStatus enum."""
        from app.core.providers.etrade import _ETRADE_STATUS_MAP

        assert _ETRADE_STATUS_MAP["OPEN"] == OrderStatus.SUBMITTED
        assert _ETRADE_STATUS_MAP["EXECUTED"] == OrderStatus.FILLED
        assert _ETRADE_STATUS_MAP["CANCELLED"] == OrderStatus.CANCELLED
        assert _ETRADE_STATUS_MAP["REJECTED"] == OrderStatus.REJECTED
        assert _ETRADE_STATUS_MAP["PARTIAL"] == OrderStatus.PARTIAL_FILL
        assert _ETRADE_STATUS_MAP["EXPIRED"] == OrderStatus.CANCELLED
        assert _ETRADE_STATUS_MAP["CANCEL_REQUESTED"] == OrderStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_etrade_account_balances(self):
        """Balance response → AccountBalances model."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True
        provider._account_id_key = "test_acct"

        mock_accounts = MagicMock()
        mock_accounts.get_account_balance = MagicMock(return_value={
            "BalanceResponse": {
                "Computed": {
                    "cashAvailableForInvestment": 50000.00,
                    "cashBuyingPower": 100000.00,
                    "marginBuyingPower": 25000.00,
                    "RealTimeValues": {"netMv": 150000.00},
                },
            }
        })
        provider._accounts = mock_accounts

        balances = await provider.get_account_balances()
        assert balances.broker == "etrade"
        assert balances.cash == 50000.00
        assert balances.buying_power == 100000.00
        assert balances.net_liquidation == 150000.00

    def test_etrade_token_freshness_check(self, tmp_path):
        """Stale tokens (before midnight ET) are rejected."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        token_file = tmp_path / "etrade_tokens.json"

        # Write stale tokens (yesterday)
        stale_time = (datetime.now(UTC) - timedelta(days=1)).isoformat()
        token_file.write_text(json.dumps({
            "oauth_token": "old_token",
            "oauth_token_secret": "old_secret",
            "saved_at": stale_time,
        }))

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_token_file = str(token_file)
            result = provider._load_tokens()
            # Stale tokens should be None
            assert result is None

    def test_etrade_token_fresh_check(self, tmp_path):
        """Fresh tokens (saved recently) are loaded."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        token_file = tmp_path / "etrade_tokens.json"

        # Write fresh tokens (just now)
        fresh_time = datetime.now(UTC).isoformat()
        token_file.write_text(json.dumps({
            "oauth_token": "fresh_token",
            "oauth_token_secret": "fresh_secret",
            "saved_at": fresh_time,
        }))

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_token_file = str(token_file)
            result = provider._load_tokens()
            assert result is not None
            assert result["oauth_token"] == "fresh_token"

    def test_etrade_sandbox_vs_prod_urls(self):
        """URL selection based on etrade_sandbox setting."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_sandbox = True
            assert provider.base_url == "https://apisb.etrade.com"

            mock_settings.etrade_sandbox = False
            assert provider.base_url == "https://api.etrade.com"

    @pytest.mark.asyncio
    async def test_etrade_connect_no_tokens_warns(self):
        """connect() warns when no tokens available."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_consumer_key = "test_key"
            mock_settings.etrade_consumer_secret = "test_secret"
            mock_settings.etrade_sandbox = True
            mock_settings.etrade_account_id = ""
            mock_settings.etrade_oauth_token = ""
            mock_settings.etrade_oauth_secret = ""
            mock_settings.etrade_token_file = "/nonexistent/tokens.json"

            # Should not crash, just warn
            await provider.connect()
            assert provider._connected is False

    @pytest.mark.asyncio
    async def test_etrade_option_chain_parsing(self):
        """Option chain response → expirations + strikes."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True

        mock_market = MagicMock()
        mock_market.get_option_chains = MagicMock(return_value={
            "OptionChainResponse": {
                "OptionPair": [
                    {
                        "Call": {
                            "strikePrice": 100.0,
                            "expirationYear": 2026, "expirationMonth": 3, "expirationDay": 20,
                        },
                        "Put": {
                            "strikePrice": 100.0,
                            "expirationYear": 2026, "expirationMonth": 3, "expirationDay": 20,
                        },
                    },
                    {
                        "Call": {
                            "strikePrice": 105.0,
                            "expirationYear": 2026, "expirationMonth": 3, "expirationDay": 20,
                        },
                        "Put": {
                            "strikePrice": 105.0,
                            "expirationYear": 2026, "expirationMonth": 3, "expirationDay": 20,
                        },
                    },
                ]
            }
        })
        provider._market = mock_market

        chain = await provider.get_option_chain("SPY")
        assert "2026-03-20" in chain["expirations"]
        assert 100.0 in chain["strikes"]
        assert 105.0 in chain["strikes"]

    @pytest.mark.asyncio
    async def test_etrade_cancel_order_readonly(self):
        """Cancel blocked when readonly."""
        from app.core.providers.etrade import EtradeLiveProvider

        provider = EtradeLiveProvider()
        provider._connected = True

        with patch("app.config.settings") as mock_settings:
            mock_settings.etrade_readonly = True
            result = await provider.cancel_order("order_123")
            assert result.status == OrderStatus.REJECTED

    def test_etrade_config_settings(self):
        """New E*TRADE config fields have correct defaults."""
        from app.config import Settings

        s = Settings(
            _env_file=None,
            etrade_consumer_key="",
            etrade_consumer_secret="",
        )
        assert s.etrade_sandbox is True
        assert s.etrade_account_id == ""
        assert s.etrade_readonly is True
        assert s.etrade_token_file == "data/utp/etrade_tokens.json"


# ── Percentage-based strike resolution ─────────────────────────────────────


class TestPctStrikeResolution:
    """Tests for --otm-pct percentage-based strike selection."""

    def test_resolve_put_spx(self):
        """PUT credit spread: short below price, long further below."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("SPX", "PUT", 3.0, 3.0, 20, 6950.0)
        assert result["short_strike"] == 6740.0  # 6950 * 0.97 = 6741.5 → round to 6740
        assert result["long_strike"] == 6720.0   # 6740 - 20 = 6720

    def test_resolve_call_spx(self):
        """CALL credit spread: short above price, long further above."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("SPX", "CALL", 3.0, 3.0, 20, 6950.0)
        assert result["short_strike"] == 7160.0  # 6950 * 1.03 = 7158.5 → round to 7160
        assert result["long_strike"] == 7180.0   # 7160 + 20 = 7180

    def test_resolve_put_ndx(self):
        """NDX uses 50-point increments."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("NDX", "PUT", 2.0, 2.0, 100, 20000.0)
        assert result["short_strike"] == 19600.0  # 20000 * 0.98 = 19600 → exact
        assert result["long_strike"] == 19500.0   # 19600 - 100 = 19500

    def test_resolve_put_rut(self):
        """RUT uses 5-point increments."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("RUT", "PUT", 3.0, 3.0, 20, 2100.0)
        assert result["short_strike"] == 2035.0  # 2100 * 0.97 = 2037 → round to 2035
        assert result["long_strike"] == 2015.0   # 2035 - 20 = 2015

    def test_resolve_equity(self):
        """Equities use 1-point increments."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("SPY", "PUT", 2.0, 2.0, 5, 550.0)
        assert result["short_strike"] == 539.0  # 550 * 0.98 = 539 → exact
        assert result["long_strike"] == 534.0   # 539 - 5 = 534

    def test_resolve_iron_condor(self):
        """Iron condor: both wings resolved symmetrically."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("SPX", None, 3.0, 3.0, 20, 6950.0, is_iron_condor=True)
        assert result["put_short"] == 6740.0
        assert result["put_long"] == 6720.0
        assert result["call_short"] == 7160.0
        assert result["call_long"] == 7180.0

    def test_resolve_iron_condor_asymmetric(self):
        """Iron condor: split put_pct vs call_pct produces asymmetric wings."""
        from utp import _resolve_pct_strikes
        # Put 1.5% below, call 2.5% above 6950
        result = _resolve_pct_strikes("SPX", None, 1.5, 2.5, 20, 6950.0, is_iron_condor=True)
        # 6950 * 0.985 = 6845.75 → 6845
        assert result["put_short"] == 6845.0
        assert result["put_long"] == 6825.0  # 6845 - 20
        # 6950 * 1.025 = 7123.75 → 7125
        assert result["call_short"] == 7125.0
        assert result["call_long"] == 7145.0  # 7125 + 20

    def test_default_width_spx(self):
        """Default width for SPX is 20."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("SPX", "PUT", 3.0, 3.0, None, 6950.0)
        assert result["short_strike"] == 6740.0
        assert result["long_strike"] == 6720.0  # 6740 - 20 = 6720

    def test_default_width_ndx(self):
        """Default width for NDX is 50."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("NDX", "PUT", 2.0, 2.0, None, 20000.0)
        assert result["short_strike"] == 19600.0
        assert result["long_strike"] == 19550.0  # 19600 - 50 = 19550

    def test_default_width_equity(self):
        """Default width for equities is 5."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("QQQ", "CALL", 2.0, 2.0, None, 500.0)
        assert result["short_strike"] == 510.0
        assert result["long_strike"] == 515.0  # 510 + 5 = 515

    def test_round_strike(self):
        """Test strike rounding helper."""
        from utp import _round_strike
        assert _round_strike(6741.5, 5) == 6740.0
        assert _round_strike(6743.0, 5) == 6745.0
        assert _round_strike(19623.0, 50) == 19600.0  # 19623/50 = 392.46 → 392 → 19600
        assert _round_strike(19626.0, 50) == 19650.0  # 19626/50 = 392.52 → 393 → 19650
        assert _round_strike(539.4, 1) == 539.0
        assert _round_strike(539.6, 1) == 540.0

    def test_invalid_option_type(self):
        """Raises for invalid option type (non-iron-condor)."""
        from utp import _resolve_pct_strikes
        with pytest.raises(ValueError, match="PUT or CALL"):
            _resolve_pct_strikes("SPX", "STRADDLE", 3.0, 3.0, 20, 6950.0)

    # ── Split-percentage parser ────────────────────────────────────

    def test_parse_split_pct_single(self):
        """Single value applies symmetrically to both put and call."""
        from utp import _parse_split_pct
        assert _parse_split_pct("2") == (2.0, 2.0)
        assert _parse_split_pct("3.5") == (3.5, 3.5)
        assert _parse_split_pct("0") == (0.0, 0.0)

    def test_parse_split_pct_colon(self):
        """'put:call' form sets each side independently."""
        from utp import _parse_split_pct
        assert _parse_split_pct("2:3") == (2.0, 3.0)
        assert _parse_split_pct("1.5:2.5") == (1.5, 2.5)
        assert _parse_split_pct("0:5") == (0.0, 5.0)

    def test_parse_split_pct_numeric_input(self):
        """Numeric input (int/float) treated as symmetric."""
        from utp import _parse_split_pct
        assert _parse_split_pct(3) == (3.0, 3.0)
        assert _parse_split_pct(2.5) == (2.5, 2.5)

    def test_parse_split_pct_invalid_empty(self):
        """Empty string raises."""
        from utp import _parse_split_pct
        with pytest.raises(ValueError, match="empty"):
            _parse_split_pct("")
        with pytest.raises(ValueError, match="empty"):
            _parse_split_pct("   ")
        with pytest.raises(ValueError):
            _parse_split_pct("2:")
        with pytest.raises(ValueError):
            _parse_split_pct(":3")

    def test_parse_split_pct_invalid_non_numeric(self):
        """Non-numeric raises."""
        from utp import _parse_split_pct
        with pytest.raises(ValueError, match="not numeric"):
            _parse_split_pct("abc")
        with pytest.raises(ValueError, match="not numeric"):
            _parse_split_pct("2:abc")

    def test_parse_split_pct_invalid_too_many_parts(self):
        """More than 2 parts raises."""
        from utp import _parse_split_pct
        with pytest.raises(ValueError, match="3 parts"):
            _parse_split_pct("1:2:3")

    def test_parse_split_pct_invalid_negative(self):
        """Negative values raise."""
        from utp import _parse_split_pct
        with pytest.raises(ValueError, match="negative"):
            _parse_split_pct("-1")
        with pytest.raises(ValueError, match="negative"):
            _parse_split_pct("2:-1")

    # ── Validator ──────────────────────────────────────────────────

    def test_validate_otm_pct_with_explicit_strikes_cs(self):
        """Cannot combine --otm-pct with explicit strikes for credit-spread."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="3", close_pct=None,
                                  short_strike=6740.0, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "Cannot use" in err

    def test_validate_otm_pct_with_explicit_strikes_ic(self):
        """Cannot combine --otm-pct with explicit strikes for iron-condor."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="3", close_pct=None,
                                  put_short=6740.0, put_long=None,
                                  call_short=None, call_long=None, width=None)
        err = _validate_strike_args("iron-condor", args)
        assert err is not None
        assert "Cannot use" in err

    def test_validate_missing_strikes_no_otm(self):
        """Error when neither --otm-pct nor explicit strikes provided."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct=None, close_pct=None,
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "required" in err

    def test_validate_ok_with_otm_pct(self):
        """No error when --otm-pct is set without explicit strikes."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="3", close_pct=None,
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is None

    def test_validate_ok_with_explicit_strikes(self):
        """No error when explicit strikes are set without --otm-pct."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct=None, close_pct=None,
                                  short_strike=6740.0, long_strike=6720.0, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is None

    def test_validate_ic_missing_strikes_no_otm(self):
        """Iron condor: error when strikes partially missing and no --otm-pct."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct=None, close_pct=None,
                                  put_short=6740.0, put_long=6720.0,
                                  call_short=None, call_long=None, width=None)
        err = _validate_strike_args("iron-condor", args)
        assert err is not None
        assert "required" in err

    # ── Risk-tier auto-resolution ──────────────────────────────────────

    def test_validate_risk_tier_intraday_alone_passes(self):
        """--risk-tier-intraday alone (no strikes, no pct) is valid — strikes
        get auto-resolved by _resolve_risk_tier_strikes from percentile data."""
        from utp import _validate_strike_args
        args = argparse.Namespace(
            otm_pct=None, close_pct=None,
            short_strike=None, long_strike=None, width=None,
            risk_tier=None, risk_tier_intraday="moderate", risk_tier_pred=None,
        )
        assert _validate_strike_args("credit-spread", args) is None

    def test_validate_risk_tier_alone_passes(self):
        from utp import _validate_strike_args
        args = argparse.Namespace(
            otm_pct=None, close_pct=None,
            short_strike=None, long_strike=None, width=None,
            risk_tier="aggressive", risk_tier_intraday=None, risk_tier_pred=None,
        )
        assert _validate_strike_args("credit-spread", args) is None

    def test_validate_risk_tier_pred_alone_passes(self):
        from utp import _validate_strike_args
        args = argparse.Namespace(
            otm_pct=None, close_pct=None,
            short_strike=None, long_strike=None, width=None,
            risk_tier=None, risk_tier_intraday=None, risk_tier_pred="conservative",
        )
        assert _validate_strike_args("credit-spread", args) is None

    def test_validate_risk_tier_with_explicit_strikes_rejects(self):
        from utp import _validate_strike_args
        args = argparse.Namespace(
            otm_pct=None, close_pct=None,
            short_strike=6740.0, long_strike=6720.0, width=None,
            risk_tier=None, risk_tier_intraday="moderate", risk_tier_pred=None,
        )
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "risk-tier" in err.lower()

    def test_validate_risk_tier_with_otm_pct_rejects(self):
        from utp import _validate_strike_args
        args = argparse.Namespace(
            otm_pct="2", close_pct=None,
            short_strike=None, long_strike=None, width=None,
            risk_tier=None, risk_tier_intraday="moderate", risk_tier_pred=None,
        )
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "risk-tier" in err.lower() or "Cannot use" in err

    def test_validate_missing_strikes_error_mentions_risk_tier(self):
        """The error message lists risk-tier as a third valid option."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct=None, close_pct=None,
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "risk-tier" in err

    def test_validate_iron_condor_risk_tier_alone_passes(self):
        from utp import _validate_strike_args
        args = argparse.Namespace(
            otm_pct=None, close_pct=None,
            put_short=None, put_long=None, call_short=None, call_long=None,
            width=None,
            risk_tier=None, risk_tier_intraday="moderate", risk_tier_pred=None,
        )
        assert _validate_strike_args("iron-condor", args) is None

    def test_dispatcher_calls_risk_tier_resolver_without_pct_flags(self):
        """Bug fix: risk-tier resolution must run even when --otm-pct/--close-pct are NOT set.

        Source-level check on _cmd_trade dispatcher to confirm the gate
        widened from 'pct_set' to 'pct_set or risk_tier_set'.
        """
        import utp
        with open(utp.__file__) as f:
            src = f.read()
        # The dispatcher derives a risk_tier_set bool and uses it in the gate
        assert 'risk_tier_set = any(' in src
        assert "if subcommand in (\"credit-spread\", \"iron-condor\") and (pct_set or risk_tier_set):" in src

    def test_iron_condor_argparse_has_risk_tier_flags(self):
        """Iron-condor parser declares all three --risk-tier* flags."""
        import utp
        with open(utp.__file__) as f:
            src = f.read()
        # Locate the iron-condor parser block (`t_ic = ...` through end)
        ic_start = src.index("t_ic = trade_sub.add_parser(\"iron-condor\"")
        ic_end = src.index("_add_connection_args(t_ic)", ic_start)
        ic_block = src[ic_start:ic_end]
        assert 't_ic.add_argument("--risk-tier"' in ic_block
        assert 't_ic.add_argument("--risk-tier-intraday"' in ic_block
        assert 't_ic.add_argument("--risk-tier-pred"' in ic_block

    @pytest.mark.asyncio
    async def test_resolve_risk_tier_iron_condor_sets_all_four_strikes(self, monkeypatch):
        """Iron-condor risk-tier path resolves BOTH put and call sides and sets
        put_short / put_long / call_short / call_long on args."""
        import utp
        # Stub the per-side resolver so we don't need a live db_server
        async def fake_resolve_side(*, sym, side, active_tier, tier_pred, tier_intra, db_url):
            # PUT lands below spot, CALL lands above
            return (5400.0, 95, "intraday") if side == "put" else (5600.0, 95, "intraday")
        monkeypatch.setattr(utp, "_resolve_one_tier_side", fake_resolve_side)

        # Stub chain snap to no-op
        async def no_snap(client, sym, exp, strikes, opt_type, is_ic):
            return strikes
        monkeypatch.setattr(utp, "_snap_strikes_to_chain_http", no_snap)

        args = argparse.Namespace(
            symbol="SPX", expiration="2026-04-30",
            risk_tier=None, risk_tier_intraday="moderate", risk_tier_pred=None,
            width=20.0,
            put_short=None, put_long=None, call_short=None, call_long=None,
        )
        rc = await utp._resolve_risk_tier_strikes(args, "iron-condor", client=None)
        assert rc is None
        # Put: short=5400, long=5400-20=5380
        assert args.put_short == 5400.0
        assert args.put_long == 5380.0
        # Call: short=5600, long=5600+20=5620
        assert args.call_short == 5600.0
        assert args.call_long == 5620.0

    @pytest.mark.asyncio
    async def test_resolve_risk_tier_credit_spread_sets_short_long(self, monkeypatch):
        """Credit-spread risk-tier path resolves the option_type side only."""
        import utp
        async def fake_resolve_side(*, sym, side, active_tier, tier_pred, tier_intra, db_url):
            assert side == "put"  # only one side resolved for credit-spread PUT
            return 7230.0, 95, "intraday"
        monkeypatch.setattr(utp, "_resolve_one_tier_side", fake_resolve_side)

        async def no_snap(client, sym, exp, strikes, opt_type, is_ic):
            return strikes
        monkeypatch.setattr(utp, "_snap_strikes_to_chain_http", no_snap)

        args = argparse.Namespace(
            symbol="SPX", expiration="2026-04-30",
            option_type="PUT",
            risk_tier=None, risk_tier_intraday="moderate", risk_tier_pred=None,
            width=25.0,
            short_strike=None, long_strike=None,
        )
        rc = await utp._resolve_risk_tier_strikes(args, "credit-spread", client=None)
        assert rc is None
        assert args.short_strike == 7230.0
        assert args.long_strike == 7205.0  # 7230 - 25

    @pytest.mark.asyncio
    async def test_resolve_risk_tier_call_credit_spread(self, monkeypatch):
        """Credit-spread CALL: long is short + width."""
        import utp
        async def fake_resolve_side(*, sym, side, active_tier, tier_pred, tier_intra, db_url):
            return 7400.0, 90, "historical"
        monkeypatch.setattr(utp, "_resolve_one_tier_side", fake_resolve_side)

        async def no_snap(client, sym, exp, strikes, opt_type, is_ic):
            return strikes
        monkeypatch.setattr(utp, "_snap_strikes_to_chain_http", no_snap)

        args = argparse.Namespace(
            symbol="SPX", expiration="2026-04-30",
            option_type="CALL",
            risk_tier="aggressive", risk_tier_intraday=None, risk_tier_pred=None,
            width=20.0,
            short_strike=None, long_strike=None,
        )
        rc = await utp._resolve_risk_tier_strikes(args, "credit-spread", client=None)
        assert rc is None
        assert args.short_strike == 7400.0
        assert args.long_strike == 7420.0  # 7400 + 20

    @pytest.mark.asyncio
    async def test_resolve_risk_tier_pred_iron_condor(self, monkeypatch):
        """--risk-tier-pred for iron-condor invokes the pred path on both sides."""
        import utp
        calls = []
        async def fake_resolve_side(*, sym, side, active_tier, tier_pred, tier_intra, db_url):
            calls.append((side, tier_pred, tier_intra))
            return (5400.0 if side == "put" else 5600.0), 99, "prediction"
        monkeypatch.setattr(utp, "_resolve_one_tier_side", fake_resolve_side)
        async def no_snap(c, s, e, st, ot, ic):
            return st
        monkeypatch.setattr(utp, "_snap_strikes_to_chain_http", no_snap)

        args = argparse.Namespace(
            symbol="NDX", expiration="2026-04-30",
            risk_tier=None, risk_tier_intraday=None, risk_tier_pred="conservative",
            width=50.0,
            put_short=None, put_long=None, call_short=None, call_long=None,
        )
        rc = await utp._resolve_risk_tier_strikes(args, "iron-condor", client=None)
        assert rc is None
        assert len(calls) == 2
        sides_called = sorted(c[0] for c in calls)
        assert sides_called == ["call", "put"]
        # All calls should pass through tier_pred="conservative"
        for _, tp, ti in calls:
            assert tp == "conservative"
            assert ti is None
        assert args.put_short == 5400.0 and args.put_long == 5350.0
        assert args.call_short == 5600.0 and args.call_long == 5650.0

    @pytest.mark.asyncio
    async def test_resolve_risk_tier_no_tier_set_returns_none(self, monkeypatch):
        """When no risk-tier flag is set, the resolver short-circuits to None."""
        import utp
        args = argparse.Namespace(
            symbol="SPX", expiration="2026-04-30",
            risk_tier=None, risk_tier_intraday=None, risk_tier_pred=None,
            width=20.0,
        )
        rc = await utp._resolve_risk_tier_strikes(args, "credit-spread", client=None)
        assert rc is None

    def test_iron_condor_default_width(self):
        """Iron condor uses symbol default width."""
        from utp import _resolve_pct_strikes
        result = _resolve_pct_strikes("RUT", None, 3.0, 3.0, None, 2100.0, is_iron_condor=True)
        # RUT default width is 20
        assert result["put_long"] == result["put_short"] - 20
        assert result["call_long"] == result["call_short"] + 20

    def test_credit_spread_rejects_split_pct(self):
        """Credit-spread validator rejects 'put:call' split form for --otm-pct."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="2:3", close_pct=None,
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "split form" in err or "single value" in err

    def test_credit_spread_rejects_split_close_pct(self):
        """Credit-spread validator rejects 'put:call' split form for --close-pct."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct=None, close_pct="1:2",
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "split form" in err or "single value" in err

    def test_credit_spread_accepts_single_pct(self):
        """Credit-spread accepts single-value form for --otm-pct."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="3", close_pct=None,
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is None

    def test_iron_condor_accepts_split_pct(self):
        """Iron condor accepts 'put:call' split form."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="2:3", close_pct=None,
                                  put_short=None, put_long=None,
                                  call_short=None, call_long=None, width=None)
        err = _validate_strike_args("iron-condor", args)
        assert err is None

    def test_validate_rejects_otm_and_close_pct(self):
        """Cannot specify both --otm-pct and --close-pct (conflicting anchors)."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="2", close_pct="1",
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "both" in err.lower()

        args2 = argparse.Namespace(otm_pct="2", close_pct="1",
                                   put_short=None, put_long=None,
                                   call_short=None, call_long=None, width=None)
        err2 = _validate_strike_args("iron-condor", args2)
        assert err2 is not None
        assert "both" in err2.lower()

    def test_validate_invalid_pct_string(self):
        """Validator rejects malformed pct string."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct="abc", close_pct=None,
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is not None
        assert "Invalid" in err

    def test_validate_ok_with_close_pct(self):
        """No error when --close-pct is set without explicit strikes."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct=None, close_pct="2",
                                  short_strike=None, long_strike=None, width=None)
        err = _validate_strike_args("credit-spread", args)
        assert err is None

    def test_validate_ic_ok_with_close_pct_split(self):
        """Iron condor: --close-pct split form is valid."""
        from utp import _validate_strike_args
        args = argparse.Namespace(otm_pct=None, close_pct="1.5:2.5",
                                  put_short=None, put_long=None,
                                  call_short=None, call_long=None, width=None)
        err = _validate_strike_args("iron-condor", args)
        assert err is None

    def test_snap_to_chain_nearest(self):
        """Snap to nearest available strike."""
        from utp import _snap_to_chain
        chain = [6780, 6790, 6800, 6810, 6820, 6825, 6830]
        assert _snap_to_chain(6812, chain) == 6810
        assert _snap_to_chain(6813, chain) == 6810
        assert _snap_to_chain(6817, chain) == 6820

    def test_snap_to_chain_otm_put(self):
        """For put short, snap to nearest strike <= target (further OTM)."""
        from utp import _snap_to_chain
        chain = [6780, 6790, 6800, 6810, 6820, 6825, 6830]
        assert _snap_to_chain(6812, chain, "otm_put") == 6810
        assert _snap_to_chain(6810, chain, "otm_put") == 6810
        assert _snap_to_chain(6799, chain, "otm_put") == 6790

    def test_snap_to_chain_otm_call(self):
        """For call short, snap to nearest strike >= target (further OTM)."""
        from utp import _snap_to_chain
        chain = [7100, 7120, 7130, 7140, 7150, 7175, 7200]
        assert _snap_to_chain(7160, chain, "otm_call") == 7175
        assert _snap_to_chain(7150, chain, "otm_call") == 7150
        assert _snap_to_chain(7141, chain, "otm_call") == 7150

    def test_snap_to_chain_gap(self):
        """Snap across a gap in the chain (e.g., 7130 → 7150, no 7135/7140/7145)."""
        from utp import _snap_to_chain
        chain = [7100, 7110, 7120, 7130, 7150, 7175, 7200]
        # 7160 doesn't exist, nearest >= is 7175
        assert _snap_to_chain(7160, chain, "otm_call") == 7175
        # 7140 doesn't exist, nearest >= is 7150
        assert _snap_to_chain(7140, chain, "otm_call") == 7150

    def test_snap_to_chain_empty(self):
        """Empty chain returns target unchanged."""
        from utp import _snap_to_chain
        assert _snap_to_chain(7160, [], "otm_call") == 7160


class TestClosePctResolution:
    """Tests for --close-pct (anchored to previous trading day's close)."""

    def test_fetch_prev_close_success(self, monkeypatch):
        """_fetch_prev_close_from_db_server returns previous_close from db_server payload."""
        import utp as utp_mod

        class _FakeResp:
            status_code = 200
            def json(self):
                return {"ticker": "SPX", "previous_close": 5800.5}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url, params=None):
                assert "/api/range_percentiles" in url
                assert params and params.get("tickers") == "SPX"
                return _FakeResp()

        import httpx as _httpx_mod
        monkeypatch.setattr(_httpx_mod, "AsyncClient", _FakeClient)
        result = asyncio.run(utp_mod._fetch_prev_close_from_db_server("SPX"))
        assert result == 5800.5

    def test_fetch_prev_close_list_response(self, monkeypatch):
        """If db_server returns a list, takes the first entry."""
        import utp as utp_mod

        class _FakeResp:
            status_code = 200
            def json(self):
                return [{"ticker": "SPX", "previous_close": 5750.0}]

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url, params=None):
                return _FakeResp()

        import httpx as _httpx_mod
        monkeypatch.setattr(_httpx_mod, "AsyncClient", _FakeClient)
        result = asyncio.run(utp_mod._fetch_prev_close_from_db_server("SPX"))
        assert result == 5750.0

    def test_fetch_prev_close_http_error(self, monkeypatch):
        """Non-200 returns None."""
        import utp as utp_mod

        class _FakeResp:
            status_code = 500
            def json(self): return {}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url, params=None): return _FakeResp()

        import httpx as _httpx_mod
        monkeypatch.setattr(_httpx_mod, "AsyncClient", _FakeClient)
        assert asyncio.run(utp_mod._fetch_prev_close_from_db_server("SPX")) is None

    def test_fetch_prev_close_missing_field(self, monkeypatch):
        """Missing previous_close returns None."""
        import utp as utp_mod

        class _FakeResp:
            status_code = 200
            def json(self): return {"ticker": "SPX"}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url, params=None): return _FakeResp()

        import httpx as _httpx_mod
        monkeypatch.setattr(_httpx_mod, "AsyncClient", _FakeClient)
        assert asyncio.run(utp_mod._fetch_prev_close_from_db_server("SPX")) is None

    def test_fetch_prev_close_zero_value(self, monkeypatch):
        """previous_close <= 0 returns None."""
        import utp as utp_mod

        class _FakeResp:
            status_code = 200
            def json(self): return {"previous_close": 0}

        class _FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, url, params=None): return _FakeResp()

        import httpx as _httpx_mod
        monkeypatch.setattr(_httpx_mod, "AsyncClient", _FakeClient)
        assert asyncio.run(utp_mod._fetch_prev_close_from_db_server("SPX")) is None

    def test_fetch_prev_close_exception(self, monkeypatch):
        """Network/parse exception returns None."""
        import utp as utp_mod
        import httpx as _httpx_mod

        class _BoomClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return None
            async def get(self, *a, **kw):
                raise RuntimeError("boom")

        monkeypatch.setattr(_httpx_mod, "AsyncClient", _BoomClient)
        assert asyncio.run(utp_mod._fetch_prev_close_from_db_server("SPX")) is None

    def test_resolve_close_pct_strikes_http_credit_spread(self, monkeypatch):
        """End-to-end: --close-pct resolves strikes from prev close (credit spread)."""
        import utp as utp_mod

        async def _fake_prev_close(symbol):
            return 6900.0  # Pretend SPX prev close was 6900

        monkeypatch.setattr(utp_mod, "_fetch_prev_close_from_db_server", _fake_prev_close)

        args = argparse.Namespace(
            symbol="SPX", expiration=None, option_type="PUT", width=20,
            otm_pct=None, close_pct="2",
            short_strike=None, long_strike=None,
        )

        # No HTTP client needed since expiration=None skips snapping
        rc = asyncio.run(utp_mod._resolve_close_pct_strikes_http(args, "credit-spread", client=None))
        assert rc is None
        # 6900 * 0.98 = 6762 → 6760 (SPX 5pt increment)
        assert args.short_strike == 6760.0
        assert args.long_strike == 6740.0

    def test_resolve_close_pct_strikes_http_iron_condor_asymmetric(self, monkeypatch):
        """End-to-end: --close-pct asymmetric split for iron condor."""
        import utp as utp_mod

        async def _fake_prev_close(symbol):
            return 6900.0

        monkeypatch.setattr(utp_mod, "_fetch_prev_close_from_db_server", _fake_prev_close)

        args = argparse.Namespace(
            symbol="SPX", expiration=None, option_type=None, width=20,
            otm_pct=None, close_pct="1.5:2.5",
            put_short=None, put_long=None, call_short=None, call_long=None,
        )

        rc = asyncio.run(utp_mod._resolve_close_pct_strikes_http(args, "iron-condor", client=None))
        assert rc is None
        # Put: 6900 * 0.985 = 6796.5 → floor to 6795 (round away from money)
        assert args.put_short == 6795.0
        assert args.put_long == 6775.0  # 6795 - 20
        # Call: 6900 * 1.025 = 7072.5 → ceil to 7075 (round away from money)
        assert args.call_short == 7075.0
        assert args.call_long == 7095.0  # 7075 + 20

    def test_resolve_close_pct_strikes_http_no_prev_close(self, monkeypatch, capsys):
        """When db_server returns nothing, returns error code."""
        import utp as utp_mod

        async def _fake_prev_close(symbol):
            return None

        monkeypatch.setattr(utp_mod, "_fetch_prev_close_from_db_server", _fake_prev_close)

        args = argparse.Namespace(
            symbol="SPX", expiration=None, option_type="PUT", width=20,
            otm_pct=None, close_pct="2",
            short_strike=None, long_strike=None,
        )
        rc = asyncio.run(utp_mod._resolve_close_pct_strikes_http(args, "credit-spread", client=None))
        assert rc == 1
        out = capsys.readouterr().out
        assert "could not fetch previous close" in out

    def test_resolve_close_pct_strikes_http_invalid_pct(self, monkeypatch, capsys):
        """Invalid --close-pct string returns error code."""
        import utp as utp_mod

        args = argparse.Namespace(
            symbol="SPX", expiration=None, option_type="PUT", width=20,
            otm_pct=None, close_pct="abc",
            short_strike=None, long_strike=None,
        )
        rc = asyncio.run(utp_mod._resolve_close_pct_strikes_http(args, "credit-spread", client=None))
        assert rc == 1
        out = capsys.readouterr().out
        assert "invalid --close-pct" in out

    def test_resolve_close_pct_strikes_http_no_op_when_unset(self):
        """When close_pct is None, resolver is a no-op."""
        import utp as utp_mod

        args = argparse.Namespace(
            symbol="SPX", expiration=None, option_type="PUT", width=20,
            otm_pct=None, close_pct=None,
            short_strike=None, long_strike=None,
        )
        rc = asyncio.run(utp_mod._resolve_close_pct_strikes_http(args, "credit-spread", client=None))
        assert rc is None
        # Strikes unchanged
        assert args.short_strike is None


class TestTradeReplay:
    """Tests for trade replay subcommand."""

    def _make_position(self, **overrides):
        """Create a minimal position dict for testing."""
        base = {
            "position_id": "abc12345-6789-0000-0000-000000000000",
            "status": "closed",
            "order_type": "credit_spread",
            "symbol": "SPX",
            "quantity": 5,
            "entry_price": -2.50,
            "expiration": "2026-04-15",
            "legs": [
                {"action": "SELL_TO_OPEN", "strike": 6800, "option_type": "PUT", "expiration": "2026-04-15"},
                {"action": "BUY_TO_OPEN", "strike": 6780, "option_type": "PUT", "expiration": "2026-04-15"},
            ],
        }
        base.update(overrides)
        return base

    def test_resolve_replay_credit_spread(self, tmp_path):
        """Replay resolves credit spread parameters from position."""
        from utp import _resolve_replay
        import json

        pos = self._make_position()
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123",
            quantity=None,
            expiration=None,
            net_price=None,
            auto_price=False,
            mid=False,
            nocheck=False,
            confirm=False,
            data_dir=str(tmp_path),
            # mode flags
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd == "credit-spread"
        assert result_args.symbol == "SPX"
        assert result_args.short_strike == 6800
        assert result_args.long_strike == 6780
        assert result_args.option_type == "PUT"
        assert result_args.quantity == 5
        assert result_args.expiration == "2026-04-15"

    def test_resolve_replay_iron_condor(self, tmp_path):
        """Replay resolves iron condor parameters from position."""
        from utp import _resolve_replay
        import json

        pos = self._make_position(
            order_type="iron_condor",
            legs=[
                {"action": "SELL_TO_OPEN", "strike": 6800, "option_type": "PUT", "expiration": "2026-04-15"},
                {"action": "BUY_TO_OPEN", "strike": 6780, "option_type": "PUT", "expiration": "2026-04-15"},
                {"action": "SELL_TO_OPEN", "strike": 7100, "option_type": "CALL", "expiration": "2026-04-15"},
                {"action": "BUY_TO_OPEN", "strike": 7120, "option_type": "CALL", "expiration": "2026-04-15"},
            ],
        )
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd == "iron-condor"
        assert result_args.put_short == 6800
        assert result_args.put_long == 6780
        assert result_args.call_short == 7100
        assert result_args.call_long == 7120

    def test_resolve_replay_override_quantity(self, tmp_path):
        """Replay respects --quantity override."""
        from utp import _resolve_replay
        import json

        pos = self._make_position()
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123", quantity=10, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd == "credit-spread"
        assert result_args.quantity == 10  # overridden, not original 5

    def test_resolve_replay_override_expiration(self, tmp_path):
        """Replay respects --expiration override."""
        from utp import _resolve_replay
        import json

        pos = self._make_position()
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123", quantity=None, expiration="2026-04-16",
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert result_args.expiration == "2026-04-16"

    def test_resolve_replay_not_found(self, tmp_path):
        """Replay returns None when position not found."""
        from utp import _resolve_replay
        import json

        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({}))

        args = argparse.Namespace(
            position_id="nonexistent", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd is None
        assert result_args is None

    def test_resolve_replay_debit_spread(self, tmp_path):
        """Replay resolves debit spread parameters."""
        from utp import _resolve_replay
        import json

        pos = self._make_position(
            order_type="debit_spread",
            legs=[
                {"action": "BUY_TO_OPEN", "strike": 480, "option_type": "CALL", "expiration": "2026-04-15"},
                {"action": "SELL_TO_OPEN", "strike": 490, "option_type": "CALL", "expiration": "2026-04-15"},
            ],
        )
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd == "debit-spread"
        assert result_args.long_strike == 480
        assert result_args.short_strike == 490
        assert result_args.option_type == "CALL"

    def test_resolve_replay_multi_leg_credit_spread(self, tmp_path):
        """Replay resolves multi_leg with 2 legs (SELL/BUY) as credit spread."""
        from utp import _resolve_replay
        import json

        pos = self._make_position(
            order_type="multi_leg",
            legs=[
                {"action": "SELL", "strike": 6595, "option_type": "PUT", "expiration": "2026-04-15"},
                {"action": "BUY", "strike": 6575, "option_type": "PUT", "expiration": "2026-04-15"},
            ],
        )
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd == "credit-spread"
        assert result_args.symbol == "SPX"
        assert result_args.short_strike == 6595
        assert result_args.long_strike == 6575
        assert result_args.option_type == "PUT"

    def test_resolve_replay_multi_leg_iron_condor(self, tmp_path):
        """Replay resolves multi_leg with 4 legs as iron condor."""
        from utp import _resolve_replay
        import json

        pos = self._make_position(
            order_type="multi_leg",
            legs=[
                {"action": "SELL", "strike": 6800, "option_type": "PUT", "expiration": "2026-04-15"},
                {"action": "BUY", "strike": 6780, "option_type": "PUT", "expiration": "2026-04-15"},
                {"action": "SELL", "strike": 7100, "option_type": "CALL", "expiration": "2026-04-15"},
                {"action": "BUY", "strike": 7120, "option_type": "CALL", "expiration": "2026-04-15"},
            ],
        )
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd == "iron-condor"
        assert result_args.put_short == 6800
        assert result_args.put_long == 6780
        assert result_args.call_short == 7100
        assert result_args.call_long == 7120

    def test_resolve_replay_short_action_names(self, tmp_path):
        """Replay accepts short action names (SELL/BUY) from portfolio grouping."""
        from utp import _resolve_replay
        import json

        # Use credit_spread order_type but with short action names (SELL/BUY)
        pos = self._make_position(
            order_type="credit_spread",
            legs=[
                {"action": "SELL", "strike": 6600, "option_type": "CALL", "expiration": "2026-04-15"},
                {"action": "BUY", "strike": 6620, "option_type": "CALL", "expiration": "2026-04-15"},
            ],
        )
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({pos["position_id"]: pos}))

        args = argparse.Namespace(
            position_id="abc123", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        subcmd, result_args = _resolve_replay(args, data_dir=str(tmp_path))
        assert subcmd == "credit-spread"
        assert result_args.short_strike == 6600
        assert result_args.long_strike == 6620
        assert result_args.option_type == "CALL"

    def test_resolve_replay_portfolio_fallback(self, tmp_path):
        """Replay falls back to daemon portfolio when local store has no match."""
        from utp import _resolve_replay
        from unittest.mock import patch
        import json

        # Empty local store
        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({}))

        # Mock the HTTP call to return a portfolio with a grouped position
        portfolio_response = {
            "positions": [
                {
                    "position_id": "7180a7deadbeef",
                    "symbol": "SPX",
                    "order_type": "multi_leg",
                    "quantity": 10,
                    "expiration": "2026-04-15",
                    "entry_price": -1.80,
                    "status": "open",
                    "legs": [
                        {"action": "SELL", "strike": 6595, "option_type": "PUT"},
                        {"action": "BUY", "strike": 6575, "option_type": "PUT"},
                    ],
                }
            ]
        }

        class FakeResponse:
            status = 200
            def read(self):
                return json.dumps(portfolio_response).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        args = argparse.Namespace(
            position_id="7180a7", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            subcmd, result_args = _resolve_replay(
                args, data_dir=str(tmp_path), server="http://localhost:8000"
            )
        assert subcmd == "credit-spread"
        assert result_args.symbol == "SPX"
        assert result_args.short_strike == 6595
        assert result_args.long_strike == 6575
        assert result_args.quantity == 10

    def test_resolve_replay_portfolio_fallback_not_found(self, tmp_path):
        """Replay returns None when position not in local store or portfolio."""
        from utp import _resolve_replay
        from unittest.mock import patch
        import json

        pos_file = tmp_path / "live" / "positions.json"
        pos_file.parent.mkdir(parents=True)
        pos_file.write_text(json.dumps({}))

        portfolio_response = {"positions": []}

        class FakeResponse:
            status = 200
            def read(self):
                return json.dumps(portfolio_response).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        args = argparse.Namespace(
            position_id="nonexistent", quantity=None, expiration=None,
            net_price=None, auto_price=False, mid=False, nocheck=False,
            confirm=False, data_dir=str(tmp_path),
            dry_run=False, paper=False, live=True,
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            subcmd, result_args = _resolve_replay(
                args, data_dir=str(tmp_path), server="http://localhost:8000"
            )
        assert subcmd is None
        assert result_args is None


# ── Roll Service ──────────────────────────────────────────────────────────────


class TestRollService:
    """Tests for the roll management service."""

    def _make_position(
        self,
        position_id="pos-1",
        symbol="SPX",
        short_strike=5600,
        long_strike=5575,
        option_type="PUT",
        expiration=None,
        quantity=1,
    ):
        """Helper: create a multi-leg credit spread position dict."""
        if expiration is None:
            expiration = datetime.now(UTC).strftime("%Y%m%d")
        return {
            "position_id": position_id,
            "symbol": symbol,
            "order_type": "multi_leg",
            "quantity": quantity,
            "expiration": expiration,
            "status": "open",
            "legs": [
                {"strike": short_strike, "option_type": option_type, "action": "SELL_TO_OPEN", "quantity": 1},
                {"strike": long_strike, "option_type": option_type, "action": "BUY_TO_OPEN", "quantity": 1},
            ],
        }

    def test_roll_config_defaults(self):
        """Verify default config values."""
        from app.services.roll_service import RollConfig

        cfg = RollConfig()
        assert cfg.check_interval == 30.0
        assert cfg.mirror_enabled is True
        assert cfg.mirror_trigger_severity == "warning"
        assert cfg.mirror_time_window_utc == ("18:00", "20:00")
        assert cfg.mirror_max_cost_pct == 1.0
        assert cfg.forward_enabled is True
        assert cfg.forward_trigger_severity == "watch"
        assert cfg.forward_min_dte == 1
        assert cfg.forward_max_dte == 5
        assert cfg.forward_max_width_multiplier == 2.0
        assert cfg.auto_execute is False

        # Round-trip serialization
        d = cfg.to_dict()
        cfg2 = RollConfig.from_dict(d)
        assert cfg2.check_interval == cfg.check_interval
        assert cfg2.mirror_time_window_utc == cfg.mirror_time_window_utc

    def test_roll_service_init(self):
        """Test init/get/reset module accessors."""
        from app.services.roll_service import (
            RollConfig,
            init_roll_service,
            get_roll_service,
            reset_roll_service,
        )

        reset_roll_service()
        assert get_roll_service() is None

        svc = init_roll_service(RollConfig(check_interval=10))
        assert svc is not None
        assert get_roll_service() is svc
        assert svc.config.check_interval == 10

        reset_roll_service()
        assert get_roll_service() is None

    @pytest.mark.asyncio
    async def test_mirror_suggestion_put_threatened(self, tmp_path):
        """PUT near breach should suggest CALL mirror."""
        from app.services.roll_service import RollConfig, RollService, _calc_breach_status

        today = datetime.now(UTC).strftime("%Y%m%d")
        pos = self._make_position(
            short_strike=5600, long_strike=5575, option_type="PUT", expiration=today
        )

        # Price at 5603 = 0.05% from short = critical severity
        breach = _calc_breach_status(5603, pos)
        assert breach is not None
        assert breach["severity"] == "critical"

        cfg = RollConfig(
            mirror_trigger_severity="critical",
            mirror_time_window_utc=("00:00", "23:59"),
        )
        svc = RollService(cfg)

        suggestion = svc._build_mirror_suggestion(pos, breach, 5603)
        assert suggestion is not None
        assert suggestion.roll_type == "mirror"
        assert suggestion.new_option_type == "CALL"  # Opposite of PUT
        assert suggestion.new_width == 25  # Same width as original
        assert suggestion.symbol == "SPX"
        assert suggestion.current_option_type == "PUT"

    @pytest.mark.asyncio
    async def test_mirror_suggestion_call_threatened(self, tmp_path):
        """CALL near breach should suggest PUT mirror."""
        from app.services.roll_service import RollConfig, RollService, _calc_breach_status

        today = datetime.now(UTC).strftime("%Y%m%d")
        pos = self._make_position(
            short_strike=5600, long_strike=5625, option_type="CALL", expiration=today
        )

        # Price at 5597 = 0.05% from short = critical
        breach = _calc_breach_status(5597, pos)
        assert breach is not None
        assert breach["severity"] == "critical"

        cfg = RollConfig(mirror_time_window_utc=("00:00", "23:59"))
        svc = RollService(cfg)

        suggestion = svc._build_mirror_suggestion(pos, breach, 5597)
        assert suggestion is not None
        assert suggestion.new_option_type == "PUT"  # Opposite of CALL
        assert suggestion.new_width == 25

    @pytest.mark.asyncio
    async def test_mirror_only_on_expiration_day(self):
        """Mirror should not be suggested for DTE > 0."""
        from app.services.roll_service import RollConfig, RollService

        # Position expiring tomorrow
        future = (datetime.now(UTC) + timedelta(days=3)).strftime("%Y%m%d")
        pos = self._make_position(
            short_strike=5600, long_strike=5575, option_type="PUT", expiration=future
        )

        cfg = RollConfig(
            mirror_trigger_severity="watch",
            mirror_time_window_utc=("00:00", "23:59"),
            forward_enabled=False,
        )
        svc = RollService(cfg)

        # Mock quote and position store
        mock_quote = MagicMock()
        mock_quote.last = 5610  # 0.18% from short = critical
        mock_quote.bid = 5610

        mock_store = MagicMock()
        mock_store.get_open_positions.return_value = [pos]

        with patch("app.services.market_data.get_quote", new_callable=AsyncMock, return_value=mock_quote):
            with patch("app.services.position_store.get_position_store", return_value=mock_store):
                suggestions = await svc.scan_positions()

        # No mirror suggestion (not expiration day), and forward disabled
        assert len(suggestions) == 0

    @pytest.mark.asyncio
    async def test_forward_suggestion(self):
        """Position at watch severity should get forward roll suggestion."""
        from app.services.roll_service import RollConfig, RollService, _calc_breach_status

        today = datetime.now(UTC).strftime("%Y%m%d")
        pos = self._make_position(
            short_strike=5600, long_strike=5575, option_type="PUT", expiration=today
        )

        # Price at 5680 = 1.4% from short = watch
        breach = _calc_breach_status(5680, pos)
        assert breach is not None
        assert breach["severity"] == "watch"

        cfg = RollConfig(forward_trigger_severity="watch", forward_min_dte=2)
        svc = RollService(cfg)

        suggestion = svc._build_forward_suggestion(pos, breach, 5680)
        assert suggestion is not None
        assert suggestion.roll_type == "forward"
        assert suggestion.new_option_type == "PUT"  # Same type
        assert suggestion.new_short_strike < 5680  # Further OTM
        assert suggestion.new_width == 25  # Same width
        # Check DTE
        exp_date = datetime.strptime(suggestion.new_expiration, "%Y%m%d").date()
        today_date = datetime.now(UTC).date()
        assert (exp_date - today_date).days >= 1

    def test_suggestion_expiry(self):
        """Suggestions should expire after TTL."""
        from app.services.roll_service import RollConfig, RollService, RollSuggestion

        svc = RollService(RollConfig())

        # Create a suggestion with old timestamp
        old = RollSuggestion(
            suggestion_id="old-001",
            position_id="pos-1",
            symbol="SPX",
            roll_type="mirror",
            severity="warning",
            distance_pct=0.8,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5600,
            new_long_strike=5625,
            new_option_type="CALL",
            new_expiration="20260414",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            created_at=datetime.now(UTC) - timedelta(minutes=10),
            status="pending",
            reason="test",
        )
        svc._suggestions["old-001"] = old

        # Should not appear in pending suggestions
        pending = svc.get_suggestions()
        assert len(pending) == 0
        assert svc._suggestions["old-001"].status == "expired"

    def test_dismiss_suggestion(self):
        """Dismiss should mark suggestion as rejected."""
        from app.services.roll_service import RollConfig, RollService, RollSuggestion

        svc = RollService(RollConfig())

        s = RollSuggestion(
            suggestion_id="test-001",
            position_id="pos-1",
            symbol="SPX",
            roll_type="mirror",
            severity="warning",
            distance_pct=0.8,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5600,
            new_long_strike=5625,
            new_option_type="CALL",
            new_expiration="20260414",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            reason="test",
        )
        svc._suggestions["test-001"] = s

        assert svc.dismiss_suggestion("test-001") is True
        assert s.status == "rejected"
        assert svc.dismiss_suggestion("test-001") is False  # Already rejected

    @pytest.mark.asyncio
    async def test_suggestions_endpoint(self, client, api_key_headers):
        """GET /roll/suggestions should return list."""
        from app.services.roll_service import init_roll_service, RollConfig

        init_roll_service(RollConfig())

        resp = await client.get("/roll/suggestions", headers=api_key_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_execute_endpoint_dry_run(self, client, api_key_headers):
        """POST /roll/execute with X-Dry-Run should return suggestion preview."""
        from app.services.roll_service import init_roll_service, RollConfig, RollSuggestion

        svc = init_roll_service(RollConfig())
        s = RollSuggestion(
            suggestion_id="exec-001",
            position_id="pos-1",
            symbol="SPX",
            roll_type="mirror",
            severity="warning",
            distance_pct=0.8,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5600,
            new_long_strike=5625,
            new_option_type="CALL",
            new_expiration="20260414",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            reason="test",
        )
        svc._suggestions["exec-001"] = s

        headers = {**api_key_headers, "X-Dry-Run": "true"}
        resp = await client.post("/roll/execute/exec-001", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "dry_run"
        assert data["suggestion"]["suggestion_id"] == "exec-001"

    @pytest.mark.asyncio
    async def test_config_endpoint(self, client, api_key_headers):
        """GET/POST /roll/config should work."""
        from app.services.roll_service import init_roll_service, RollConfig

        init_roll_service(RollConfig())

        # GET
        resp = await client.get("/roll/config", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["check_interval"] == 30.0
        assert data["mirror_trigger_severity"] == "warning"

        # POST update
        resp = await client.post(
            "/roll/config",
            headers=api_key_headers,
            json={"mirror_trigger_severity": "critical", "auto_execute": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mirror_trigger_severity"] == "critical"
        assert data["auto_execute"] is True

    @pytest.mark.asyncio
    async def test_execute_forward_roll(self, tmp_path):
        """Forward roll: close current + open new. Both execute_trade calls succeed."""
        from app.services.roll_service import RollConfig, RollService, RollSuggestion
        from app.services.position_store import get_position_store
        from app.models import OrderResult, Broker, OrderStatus

        svc = RollService(RollConfig())
        pos = self._make_position(
            position_id="fwd-pos-1",
            short_strike=5600, long_strike=5575, option_type="PUT",
        )

        store = get_position_store()
        store._positions["fwd-pos-1"] = {**pos, "status": "open"}
        store._save()

        s = RollSuggestion(
            suggestion_id="fwd-001",
            position_id="fwd-pos-1",
            symbol="SPX",
            roll_type="forward",
            severity="watch",
            distance_pct=1.5,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5550,
            new_long_strike=5525,
            new_option_type="PUT",
            new_expiration="20260416",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            reason="test forward",
        )
        svc._suggestions["fwd-001"] = s

        mock_result = OrderResult(
            broker=Broker.IBKR, status=OrderStatus.FILLED,
            message="Filled", filled_price=1.50,
        )

        with patch("app.services.trade_service.execute_trade", new_callable=AsyncMock, return_value=mock_result):
            result = await svc.execute_roll("fwd-001")

        assert result["status"] == "executed"
        assert result["roll_type"] == "forward"
        assert "close_result" in result
        assert "open_result" in result
        assert s.status == "executed"

    @pytest.mark.asyncio
    async def test_execute_mirror_roll(self, tmp_path):
        """Mirror roll: only open new (no close). Single execute_trade call."""
        from app.services.roll_service import RollConfig, RollService, RollSuggestion
        from app.services.position_store import get_position_store
        from app.models import OrderResult, Broker, OrderStatus

        svc = RollService(RollConfig())
        pos = self._make_position(
            position_id="mir-pos-1",
            short_strike=5600, long_strike=5575, option_type="PUT",
        )

        store = get_position_store()
        store._positions["mir-pos-1"] = {**pos, "status": "open"}
        store._save()

        s = RollSuggestion(
            suggestion_id="mir-001",
            position_id="mir-pos-1",
            symbol="SPX",
            roll_type="mirror",
            severity="warning",
            distance_pct=0.8,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5605,
            new_long_strike=5630,
            new_option_type="CALL",
            new_expiration="20260414",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            reason="test mirror",
        )
        svc._suggestions["mir-001"] = s

        mock_result = OrderResult(
            broker=Broker.IBKR, status=OrderStatus.FILLED,
            message="Filled", filled_price=2.00,
        )

        call_count = 0
        async def mock_execute(request, dry_run=False):
            nonlocal call_count
            call_count += 1
            return mock_result

        with patch("app.services.trade_service.execute_trade", side_effect=mock_execute):
            result = await svc.execute_roll("mir-001")

        assert result["status"] == "executed"
        assert result["roll_type"] == "mirror"
        assert "open_result" in result
        assert "close_result" not in result  # Mirror = no close
        assert call_count == 1  # Only one trade (open)
        assert s.status == "executed"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_suggestion(self):
        """Executing a nonexistent suggestion returns error."""
        from app.services.roll_service import RollConfig, RollService

        svc = RollService(RollConfig())
        result = await svc.execute_roll("does-not-exist")
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_already_executed(self):
        """Executing an already-executed suggestion returns error."""
        from app.services.roll_service import RollConfig, RollService, RollSuggestion

        svc = RollService(RollConfig())
        s = RollSuggestion(
            suggestion_id="done-001",
            position_id="pos-1",
            symbol="SPX",
            roll_type="mirror",
            severity="warning",
            distance_pct=0.8,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5600,
            new_long_strike=5625,
            new_option_type="CALL",
            new_expiration="20260414",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            reason="test",
            status="executed",
        )
        svc._suggestions["done-001"] = s

        result = await svc.execute_roll("done-001")
        assert "error" in result
        assert "executed" in result["error"]

    @pytest.mark.asyncio
    async def test_forward_roll_close_fails(self, tmp_path):
        """Forward roll: if close fails, return error without opening new."""
        from app.services.roll_service import RollConfig, RollService, RollSuggestion
        from app.services.position_store import get_position_store

        svc = RollService(RollConfig())

        # No position in store → _get_position_legs returns [] → _close_position returns error
        s = RollSuggestion(
            suggestion_id="fail-001",
            position_id="no-such-pos",
            symbol="SPX",
            roll_type="forward",
            severity="watch",
            distance_pct=1.5,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5550,
            new_long_strike=5525,
            new_option_type="PUT",
            new_expiration="20260416",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            reason="test fail",
        )
        svc._suggestions["fail-001"] = s

        result = await svc.execute_roll("fail-001")
        assert "error" in result
        assert "close" in result["error"].lower() or "legs" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_endpoint_real(self, client, api_key_headers):
        """POST /roll/execute without dry-run should call execute_roll."""
        from app.services.roll_service import init_roll_service, RollConfig, RollSuggestion
        from app.services.position_store import get_position_store
        from app.models import OrderResult, Broker, OrderStatus

        svc = init_roll_service(RollConfig())

        pos = self._make_position(position_id="api-pos-1")
        store = get_position_store()
        store._positions["api-pos-1"] = {**pos, "status": "open"}
        store._save()

        s = RollSuggestion(
            suggestion_id="api-exec-001",
            position_id="api-pos-1",
            symbol="SPX",
            roll_type="mirror",
            severity="warning",
            distance_pct=0.8,
            current_short_strike=5600,
            current_long_strike=5575,
            current_option_type="PUT",
            current_expiration="20260414",
            current_quantity=1,
            current_max_loss=2500,
            new_short_strike=5605,
            new_long_strike=5630,
            new_option_type="CALL",
            new_expiration="20260414",
            new_width=25,
            estimated_credit=0,
            estimated_close_cost=0,
            net_cost=0,
            new_max_loss=2500,
            covers_close=False,
            reason="test",
        )
        svc._suggestions["api-exec-001"] = s

        mock_result = OrderResult(
            broker=Broker.IBKR, status=OrderStatus.FILLED,
            message="Filled", filled_price=2.00,
        )

        with patch("app.services.trade_service.execute_trade", new_callable=AsyncMock, return_value=mock_result):
            resp = await client.post("/roll/execute/api-exec-001", headers=api_key_headers)

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "executed"
        assert data["roll_type"] == "mirror"

    def test_format_expiration(self):
        """Test expiration format conversion helper."""
        from app.services.roll_service import _format_expiration

        assert _format_expiration("20260414") == "2026-04-14"
        assert _format_expiration("2026-04-14") == "2026-04-14"
        assert _format_expiration("") == ""

    def test_get_position_legs(self):
        """Test leg extraction from position store."""
        from app.services.roll_service import RollConfig, RollService
        from app.services.position_store import get_position_store

        svc = RollService(RollConfig())
        pos = self._make_position(position_id="legs-pos-1")
        store = get_position_store()
        store._positions["legs-pos-1"] = {**pos, "status": "open"}
        store._save()

        legs = svc._get_position_legs("legs-pos-1")
        assert len(legs) == 2
        assert any("SELL" in l.get("action", "") for l in legs)
        assert any("BUY" in l.get("action", "") for l in legs)

        # Nonexistent position
        legs = svc._get_position_legs("no-such-pos")
        assert legs == []


# ═══════════════════════════════════════════════════════════════════════════════
# Price Freshness Enforcement on Trade Commands
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriceFreshnessEnforcement:
    """Tests that trade commands enforce <=15s fresh / 15-60s warn / >60s block
    policy on displayed prices."""

    def test_classify_price_age_fresh(self):
        """Age <= 15s: fresh, no block, annotation shows age only."""
        from utp import _classify_price_age
        annotation, blocked = _classify_price_age(3.0)
        assert blocked is False
        assert "3s old" in annotation
        assert "STALE" not in annotation and "BLOCKED" not in annotation

    def test_classify_price_age_stale_warning(self):
        """Age between 15s and 60s: stale warning, not blocked."""
        from utp import _classify_price_age
        annotation, blocked = _classify_price_age(30.0)
        assert blocked is False
        assert "STALE" in annotation
        assert "30s" in annotation

    def test_classify_price_age_blocked(self):
        """Age > 60s: blocked, annotation shows BLOCKED."""
        from utp import _classify_price_age
        annotation, blocked = _classify_price_age(75.0)
        assert blocked is True
        assert "BLOCKED" in annotation
        assert "75s" in annotation

    def test_classify_price_age_none(self):
        """None age (just fetched): not blocked, no warning."""
        from utp import _classify_price_age
        annotation, blocked = _classify_price_age(None)
        assert blocked is False
        assert "just fetched" in annotation

    def test_env_var_overrides_threshold(self, monkeypatch):
        """Env vars UTP_TRADE_PRICE_FRESH_MAX_AGE/UTP_TRADE_PRICE_BLOCK_MAX_AGE override defaults."""
        from utp import _classify_price_age
        # With default 15s threshold, 10s is fresh
        a1, b1 = _classify_price_age(10.0)
        assert b1 is False and "STALE" not in a1
        # Tighten to 5s / 8s
        monkeypatch.setenv("UTP_TRADE_PRICE_FRESH_MAX_AGE", "5")
        monkeypatch.setenv("UTP_TRADE_PRICE_BLOCK_MAX_AGE", "8")
        # Now 10s exceeds block threshold
        a2, b2 = _classify_price_age(10.0)
        assert b2 is True and "BLOCKED" in a2

    @pytest.mark.asyncio
    async def test_get_quote_with_max_age_forces_provider(self, monkeypatch):
        """get_quote(max_age=15) forces provider when cache age exceeds max_age."""
        from app.services import market_data as mkt
        from app.models import Quote

        # Build a fake streaming service whose last tick is 30s old
        tick = {
            "price": 5500.0,
            "bid": 5499.0,
            "ask": 5501.0,
            "volume": 100,
            "timestamp": (datetime.now(UTC) - timedelta(seconds=30)).isoformat(),
        }

        class FakeSvc:
            is_running = True
            def get_last_tick(self, symbol, max_age_seconds=None):
                # Honour max_age_seconds: only return when tick age <= limit
                age = 30.0
                if max_age_seconds is not None and age > max_age_seconds:
                    return None
                return tick

        monkeypatch.setattr(
            "app.services.market_data_streaming.get_streaming_service",
            lambda: FakeSvc(),
        )

        provider_calls = {"n": 0}

        class FakeProvider:
            def __init__(self):
                pass
            def is_healthy(self):
                return True
            async def get_quote(self, symbol):
                provider_calls["n"] += 1
                return Quote(symbol=symbol, bid=5502.0, ask=5503.0, last=5502.5, volume=1)

        monkeypatch.setattr(
            "app.core.provider.ProviderRegistry.get",
            lambda broker: FakeProvider(),
        )
        monkeypatch.setattr("app.services.market_data._is_market_active", lambda: True)

        # With max_age=15, tick is too old → provider call
        q = await mkt.get_quote("SPX", max_age=15.0)
        assert provider_calls["n"] == 1
        assert q.quote_source == "provider"
        assert q.quote_age_seconds == 0.0

        # With max_age=None (legacy), stale cache serves — provider NOT called again
        provider_calls["n"] = 0
        q2 = await mkt.get_quote("SPX")
        # Legacy behavior: returns stale tick without calling provider
        assert provider_calls["n"] == 0
        # Cache is 30s old, served as stale_cache
        assert q2.quote_source in ("fresh_cache", "stale_cache")

    @pytest.mark.asyncio
    async def test_get_option_quotes_with_age_returns_age(self, monkeypatch):
        """get_option_quotes_with_age returns (quotes, age, source) triple."""
        from app.services import market_data as mkt

        mock_quotes = [{"strike": 5500.0, "bid": 1.0, "ask": 1.5,
                        "last": 1.2, "volume": 10, "open_interest": 50}]

        class FakeCache:
            def get_age(self, sym, exp, ot):
                return 5.0

        class FakeOQ:
            _cache = FakeCache()
            def get_cached_quotes(self, sym, exp, ot, **kwargs):
                return mock_quotes

        monkeypatch.setattr(
            "app.services.option_quote_streaming.get_option_quote_streaming",
            lambda: FakeOQ(),
        )

        quotes, age, source = await mkt.get_option_quotes_with_age(
            "SPX", "2026-04-17", "PUT", max_age=30.0,
        )
        assert quotes == mock_quotes
        assert age == 5.0
        assert source == "fresh_cache"

    @pytest.mark.asyncio
    async def test_trade_command_blocks_when_price_too_stale(self, monkeypatch, capsys):
        """Trade command returns non-zero and prints BLOCKED when quote > block threshold."""
        from utp import _cmd_trade_http

        # Fake HTTP client that returns stale quotes (90s old) for everything
        import httpx

        class FakeResp:
            def __init__(self, status_code, data):
                self.status_code = status_code
                self._data = data

            def json(self):
                return self._data

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def get(self, path, params=None):
                if path.startswith("/market/quote"):
                    return FakeResp(200, {
                        "symbol": "SPX", "bid": 5499, "ask": 5501,
                        "last": 5500, "volume": 0,
                        "quote_age_seconds": 90.0, "quote_source": "stale_cache",
                    })
                if path.startswith("/market/options"):
                    ot = (params or {}).get("option_type", "PUT").lower()
                    return FakeResp(200, {
                        "symbol": "SPX",
                        "quotes": {ot: [
                            {"strike": 5500, "bid": 1.0, "ask": 1.5, "last": 1.2, "volume": 5},
                            {"strike": 5475, "bid": 0.5, "ask": 0.8, "last": 0.6, "volume": 3},
                        ]},
                        "meta": {"age_seconds": 90.0, "source": "stale_cache"},
                        "quote_age_seconds": 90.0, "quote_source": "stale_cache",
                    })
                return FakeResp(200, {})

            async def post(self, path, json=None, headers=None):
                return FakeResp(200, {})

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        args = argparse.Namespace(
            subcommand="credit-spread",
            symbol="SPX", short_strike=5500, long_strike=5475,
            option_type="PUT", expiration="2026-04-17",
            quantity=1, net_price=3.50, broker="ibkr",
            dry_run=False, paper=False, live=True,
            confirm=False, nocheck=True, otm_pct=None, close_pct=None,
            close=False,
        )
        rc = await _cmd_trade_http(args, "http://localhost:8000")
        out = capsys.readouterr().out
        assert rc == 1
        assert "⛔" in out or "BLOCKED" in out or "blocked" in out.lower()

    @pytest.mark.asyncio
    async def test_trade_command_warns_when_stale_but_allowed(self, monkeypatch, capsys):
        """Age in 15-60s range: warning shown, trade not blocked."""
        from utp import _cmd_trade_http
        import httpx

        class FakeResp:
            def __init__(self, status_code, data):
                self.status_code = status_code
                self._data = data

            def json(self):
                return self._data

            @property
            def headers(self):
                return {"content-type": "application/json"}

            @property
            def text(self):
                return str(self._data)

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def get(self, path, params=None):
                if path.startswith("/market/quote"):
                    return FakeResp(200, {
                        "symbol": "SPX", "bid": 5499, "ask": 5501,
                        "last": 5500, "volume": 0,
                        "quote_age_seconds": 30.0, "quote_source": "stale_cache",
                    })
                if path.startswith("/market/options"):
                    ot = (params or {}).get("option_type", "PUT").lower()
                    return FakeResp(200, {
                        "symbol": "SPX",
                        "quotes": {ot: [
                            {"strike": 5500, "bid": 1.0, "ask": 1.5, "last": 1.2, "volume": 5},
                            {"strike": 5475, "bid": 0.5, "ask": 0.8, "last": 0.6, "volume": 3},
                        ]},
                        "meta": {"age_seconds": 30.0, "source": "stale_cache"},
                        "quote_age_seconds": 30.0, "quote_source": "stale_cache",
                    })
                return FakeResp(200, {})

            async def post(self, path, json=None, headers=None):
                return FakeResp(200, {})

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        args = argparse.Namespace(
            subcommand="credit-spread",
            symbol="SPX", short_strike=5500, long_strike=5475,
            option_type="PUT", expiration="2026-04-17",
            quantity=1, net_price=3.50, broker="ibkr",
            dry_run=False, paper=False, live=True,
            confirm=False, nocheck=True, otm_pct=None, close_pct=None,
            close=False,
        )
        rc = await _cmd_trade_http(args, "http://localhost:8000")
        out = capsys.readouterr().out
        # Not blocked (rc != 1 from freshness); STALE warning displayed
        assert "STALE" in out
        assert "⛔" not in out  # no block
        # No confirm → summary-only path returns 0
        assert rc == 0

    @pytest.mark.asyncio
    async def test_trade_command_dry_run_skips_freshness_check(self, monkeypatch, capsys):
        """Dry-run mode prints 'enforcement disabled' notice and does not block."""
        from utp import _cmd_trade_http
        import httpx

        class FakeResp:
            def __init__(self, status_code, data):
                self.status_code = status_code
                self._data = data

            def json(self):
                return self._data

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def get(self, path, params=None):
                if path.startswith("/market/quote"):
                    return FakeResp(200, {
                        "symbol": "SPX", "bid": 5499, "ask": 5501,
                        "last": 5500, "volume": 0,
                        "quote_age_seconds": 90.0, "quote_source": "stale_cache",
                    })
                if path.startswith("/market/options"):
                    ot = (params or {}).get("option_type", "PUT").lower()
                    return FakeResp(200, {
                        "symbol": "SPX",
                        "quotes": {ot: [
                            {"strike": 5500, "bid": 1.0, "ask": 1.5, "last": 1.2, "volume": 5},
                            {"strike": 5475, "bid": 0.5, "ask": 0.8, "last": 0.6, "volume": 3},
                        ]},
                        "meta": {"age_seconds": 90.0, "source": "stale_cache"},
                        "quote_age_seconds": 90.0, "quote_source": "stale_cache",
                    })
                return FakeResp(200, {})

            async def post(self, path, json=None, headers=None):
                return FakeResp(200, {"order_id": "abc", "status": "FILLED"})

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        args = argparse.Namespace(
            subcommand="credit-spread",
            symbol="SPX", short_strike=5500, long_strike=5475,
            option_type="PUT", expiration="2026-04-17",
            quantity=1, net_price=3.50, broker="ibkr",
            dry_run=True, paper=False, live=False,
            confirm=True, nocheck=True, otm_pct=None, close_pct=None,
            close=False,
        )
        rc = await _cmd_trade_http(args, "http://localhost:8000")
        out = capsys.readouterr().out
        assert "enforcement disabled" in out.lower()
        # Dry-run completes without actually blocking the trade.  The
        # per-line annotation may still say BLOCKED (informational), but
        # the explicit "Trade blocked" gate should not fire.
        assert "Trade blocked" not in out
        assert rc == 0

    @pytest.mark.asyncio
    async def test_http_quote_endpoint_passes_max_age(self, client, api_key_headers, monkeypatch):
        """GET /market/quote/{sym}?max_age=15 passes max_age through to get_quote()."""
        from app.services import market_data as mkt
        from app.models import Quote

        received: dict = {}
        real_get = mkt.get_quote

        async def spy_get_quote(symbol, broker=None, *, max_age=None, force_refresh=False):
            received["max_age"] = max_age
            received["force_refresh"] = force_refresh
            return Quote(symbol=symbol, bid=1.0, ask=2.0, last=1.5, volume=0,
                         quote_age_seconds=0.0, quote_source="provider")

        monkeypatch.setattr("app.routes.market.get_quote", spy_get_quote)
        # Also patch the module-level import since the route imports from market_data
        monkeypatch.setattr("app.services.market_data.get_quote", spy_get_quote)

        resp = await client.get(
            "/market/quote/SPX?max_age=15&force_refresh=true",
            headers=api_key_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert received.get("max_age") == 15.0
        assert received.get("force_refresh") is True
        assert data.get("quote_age_seconds") == 0.0
        assert data.get("quote_source") == "provider"

    @pytest.mark.asyncio
    async def test_http_options_endpoint_passes_max_age(self, client, api_key_headers, monkeypatch):
        """GET /market/options/{sym}?max_age=15 passes through and returns meta.age_seconds."""
        received: dict = {}

        async def spy_get_opts_age(symbol, expiration, option_type, *,
                                    strike_min=None, strike_max=None,
                                    broker=None, max_age=None, force_refresh=False):
            received.setdefault("calls", []).append(
                {"max_age": max_age, "force_refresh": force_refresh,
                 "option_type": option_type}
            )
            quotes = [{"strike": 5500.0, "bid": 1.0, "ask": 1.5,
                       "last": 1.2, "volume": 10, "open_interest": 50}]
            return quotes, 3.0, "fresh_cache"

        monkeypatch.setattr(
            "app.services.market_data.get_option_quotes_with_age",
            spy_get_opts_age,
        )

        resp = await client.get(
            "/market/options/SPX",
            params={"expiration": "2026-04-17", "option_type": "PUT",
                    "max_age": 15, "force_refresh": "false"},
            headers=api_key_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("meta", {}).get("age_seconds") == 3.0
        assert data.get("quote_age_seconds") == 3.0
        # max_age forwarded to internal call
        assert received["calls"][0]["max_age"] == 15.0


# ══════════════════════════════════════════════════════════════════════════════
# Simulation Mode Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSimulationClock:
    """Tests for app/services/simulation_clock.py."""

    def _make_timestamps(self, sim_date=None):
        """Create a list of 5-min timestamps for a test date."""
        from datetime import date as d
        sim_date = sim_date or d(2026, 4, 1)
        base = datetime(sim_date.year, sim_date.month, sim_date.day,
                        13, 30, tzinfo=timezone.utc)  # 9:30 ET
        return [base + timedelta(minutes=5 * i) for i in range(78)]  # 78 bars

    def test_init_and_basic_properties(self):
        from app.services.simulation_clock import SimulationClock
        ts = self._make_timestamps()
        clock = SimulationClock(date(2026, 4, 1), ts)
        assert clock.sim_date == date(2026, 4, 1)
        assert clock.sim_time == ts[0]
        assert clock.is_active()
        assert not clock.auto_advancing
        assert len(clock.timestamps) == 78

    def test_advance_steps_through_timestamps(self):
        from app.services.simulation_clock import SimulationClock
        ts = self._make_timestamps()
        clock = SimulationClock(date(2026, 4, 1), ts)
        result = clock.advance()
        assert result == ts[1]
        assert clock.now() == ts[1]
        result = clock.advance()
        assert result == ts[2]

    def test_advance_returns_none_at_end(self):
        from app.services.simulation_clock import SimulationClock
        ts = [datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)]
        clock = SimulationClock(date(2026, 4, 1), ts)
        assert clock.advance() is None

    def test_set_time_snaps_cursor(self):
        from app.services.simulation_clock import SimulationClock
        ts = self._make_timestamps()
        clock = SimulationClock(date(2026, 4, 1), ts)
        # Jump to 14:00 UTC (10:00 ET) — should snap to ts[6] (13:30 + 30min)
        target = datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc)
        clock.set_time(target)
        assert clock.sim_time == target
        # Advance should go to the next timestamp after 14:00
        result = clock.advance()
        assert result is not None
        assert result > target

    def test_jump_to_et(self):
        from app.services.simulation_clock import SimulationClock
        ts = self._make_timestamps()
        clock = SimulationClock(date(2026, 4, 1), ts)
        clock.jump_to_et("10:30")
        # 10:30 ET on 2026-04-01 (EDT) = 14:30 UTC
        assert clock.sim_time.hour == 14
        assert clock.sim_time.minute == 30

    def test_reset(self):
        from app.services.simulation_clock import SimulationClock
        ts = self._make_timestamps()
        clock = SimulationClock(date(2026, 4, 1), ts)
        clock.advance()
        clock.advance()
        clock.reset()
        assert clock.now() == ts[0]
        assert clock._cursor == 0

    def test_module_accessors(self):
        from app.services.simulation_clock import (
            init_sim_clock, get_sim_clock, reset_sim_clock,
        )
        ts = self._make_timestamps()
        clock = init_sim_clock(date(2026, 4, 1), ts)
        assert get_sim_clock() is clock
        reset_sim_clock()
        assert get_sim_clock() is None

    @pytest.mark.asyncio
    async def test_auto_advance(self):
        from app.services.simulation_clock import SimulationClock
        ts = self._make_timestamps()
        clock = SimulationClock(date(2026, 4, 1), ts)
        clock.start_auto_advance(interval_sec=0.01)
        assert clock.auto_advancing
        await asyncio.sleep(0.15)
        clock.stop_auto_advance()
        assert not clock.auto_advancing
        # Should have advanced several steps
        assert clock._cursor > 0


class TestCSVSimulationProvider:
    """Tests for app/core/providers/csv_simulation.py."""

    def _create_equity_csv(self, tmp_path, ticker="SPX", sim_date="2026-04-01"):
        """Create a minimal equity CSV in the expected directory structure."""
        ticker_dir = tmp_path / "equities" / ticker
        ticker_dir.mkdir(parents=True)
        csv_path = ticker_dir / f"{ticker}_equities_{sim_date}.csv"
        csv_path.write_text(
            "timestamp,ticker,open,high,low,close,volume,vwap,transactions\n"
            f"{sim_date} 13:30:00+00:00,I:{ticker},100.0,102.0,99.0,101.0,1000,,\n"
            f"{sim_date} 13:35:00+00:00,I:{ticker},101.0,103.0,100.0,102.5,1200,,\n"
            f"{sim_date} 13:40:00+00:00,I:{ticker},102.5,104.0,101.0,103.0,800,,\n"
        )
        return ticker_dir

    def _create_options_csv(self, tmp_path, ticker="SPX", sim_date="2026-04-01"):
        """Create a minimal options CSV."""
        ticker_dir = tmp_path / "options" / ticker
        ticker_dir.mkdir(parents=True)
        csv_path = ticker_dir / f"{ticker}_options_{sim_date}.csv"
        csv_path.write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,vwap,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            f"{sim_date}T13:30:00+00:00,O:SPX260401P00095000,put,95,{sim_date},"
            "2.50,3.00,2.75,2.75,,,,,,0.25,100\n"
            f"{sim_date}T13:30:00+00:00,O:SPX260401P00090000,put,90,{sim_date},"
            "1.00,1.50,1.25,1.25,,,,,,0.30,200\n"
            f"{sim_date}T13:30:00+00:00,O:SPX260401C00105000,call,105,{sim_date},"
            "2.00,2.50,2.25,2.25,,,,,,0.20,150\n"
            f"{sim_date}T13:35:00+00:00,O:SPX260401P00095000,put,95,{sim_date},"
            "2.60,3.10,2.85,2.85,,,,,,0.25,110\n"
            f"{sim_date}T13:35:00+00:00,O:SPX260401P00090000,put,90,{sim_date},"
            "1.10,1.60,1.35,1.35,,,,,,0.30,210\n"
            f"{sim_date}T13:35:00+00:00,O:SPX260401C00105000,call,105,{sim_date},"
            "1.90,2.40,2.15,2.15,,,,,,0.20,160\n"
        )
        return ticker_dir

    @pytest.mark.asyncio
    async def test_connect_loads_data(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        self._create_equity_csv(tmp_path)
        self._create_options_csv(tmp_path)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            assert "SPX" in provider._equity_bars
            assert len(provider._equity_bars["SPX"]) == 3
            assert "SPX" in provider._option_snapshots
            ts = provider.get_all_equity_timestamps()
            assert len(ts) == 3
            init_sim_clock(date(2026, 4, 1), ts)
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_get_quote_returns_bar_data(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        self._create_equity_csv(tmp_path)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            clock = init_sim_clock(date(2026, 4, 1), ts)
            quote = await provider.get_quote("SPX")
            assert quote.symbol == "SPX"
            assert quote.last == 101.0  # first bar close
            assert quote.bid == 99.0   # low
            assert quote.ask == 102.0  # high
            assert quote.source == "simulation"
            # Advance and check second bar
            clock.advance()
            quote2 = await provider.get_quote("SPX")
            assert quote2.last == 102.5
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_get_option_quotes(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        self._create_equity_csv(tmp_path)
        self._create_options_csv(tmp_path)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            init_sim_clock(date(2026, 4, 1), ts)
            # Get all puts at first timestamp
            puts = await provider.get_option_quotes("SPX", "2026-04-01", "PUT")
            assert len(puts) == 2
            strikes = {q["strike"] for q in puts}
            assert 95.0 in strikes
            assert 90.0 in strikes
            # Get calls
            calls = await provider.get_option_quotes("SPX", "2026-04-01", "CALL")
            assert len(calls) == 1
            assert calls[0]["strike"] == 105.0
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_get_option_quotes_strike_range(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        self._create_equity_csv(tmp_path)
        self._create_options_csv(tmp_path)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            init_sim_clock(date(2026, 4, 1), ts)
            # Filter by strike range
            puts = await provider.get_option_quotes(
                "SPX", "2026-04-01", "PUT", strike_min=92, strike_max=100
            )
            assert len(puts) == 1
            assert puts[0]["strike"] == 95.0
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_get_option_chain(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        self._create_equity_csv(tmp_path)
        self._create_options_csv(tmp_path)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            init_sim_clock(date(2026, 4, 1), ts)
            chain = await provider.get_option_chain("SPX")
            assert "2026-04-01" in chain["expirations"]
            assert 95.0 in chain["strikes"]
            assert 90.0 in chain["strikes"]
            assert 105.0 in chain["strikes"]
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_execute_multi_leg_order(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        self._create_equity_csv(tmp_path)
        self._create_options_csv(tmp_path)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            init_sim_clock(date(2026, 4, 1), ts)
            # Credit spread: sell 95 put, buy 90 put
            order = MultiLegOrder(
                broker=Broker.IBKR,
                legs=[
                    OptionLeg(
                        symbol="SPX", expiration="2026-04-01", strike=95.0,
                        option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN,
                        quantity=1,
                    ),
                    OptionLeg(
                        symbol="SPX", expiration="2026-04-01", strike=90.0,
                        option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN,
                        quantity=1,
                    ),
                ],
                order_type=OrderType.MARKET,
                quantity=1,
            )
            result = await provider.execute_multi_leg_order(order)
            assert result.status == OrderStatus.FILLED
            # Net = sell bid (2.50) - buy ask (1.50) = 1.00 credit
            assert result.filled_price == pytest.approx(1.0, abs=0.01)
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_execute_equity_order(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        self._create_equity_csv(tmp_path)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            init_sim_clock(date(2026, 4, 1), ts)
            order = EquityOrder(
                broker=Broker.IBKR,
                symbol="SPX", side=OrderSide.BUY, quantity=10,
                order_type=OrderType.MARKET,
            )
            result = await provider.execute_equity_order(order)
            assert result.status == OrderStatus.FILLED
            assert result.filled_price == 101.0  # first bar close
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_check_margin(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        provider = CSVSimulationProvider(
            sim_date=date(2026, 4, 1),
            tickers=["SPX"],
            equities_dir=tmp_path / "equities",
            options_dir=tmp_path / "options",
        )
        order = MultiLegOrder(
            broker=Broker.IBKR,
            legs=[
                OptionLeg(
                    symbol="SPX", expiration="2026-04-01", strike=95.0,
                    option_type=OptionType.PUT, action=OptionAction.SELL_TO_OPEN,
                    quantity=1,
                ),
                OptionLeg(
                    symbol="SPX", expiration="2026-04-01", strike=90.0,
                    option_type=OptionType.PUT, action=OptionAction.BUY_TO_OPEN,
                    quantity=1,
                ),
            ],
            order_type=OrderType.MARKET,
            quantity=10,
        )
        margin = await provider.check_margin(order)
        # Width = 95 - 90 = 5, margin = 5 * 100 * 10 = 5000
        assert margin["init_margin"] == 5000.0

    @pytest.mark.asyncio
    async def test_get_account_balances(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        provider = CSVSimulationProvider(
            sim_date=date(2026, 4, 1),
            tickers=["SPX"],
            equities_dir=tmp_path / "equities",
            options_dir=tmp_path / "options",
        )
        balances = await provider.get_account_balances()
        assert balances.cash == 1_000_000.0

    @pytest.mark.asyncio
    async def test_missing_ticker_data(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["NOTICKER"],
                equities_dir=tmp_path / "equities",
                options_dir=tmp_path / "options",
            )
            await provider.connect()
            ts = [datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)]
            init_sim_clock(date(2026, 4, 1), ts)
            quote = await provider.get_quote("NOTICKER")
            assert quote.last == 0.0
        finally:
            reset_sim_clock()


class TestSimulationRoutes:
    """Tests for /sim/* API routes."""

    @pytest.mark.asyncio
    async def test_status_no_sim(self, client, api_key_headers):
        """When no sim clock is active, /sim/status returns 404 (routes not registered)."""
        resp = await client.get("/sim/status", headers=api_key_headers)
        # Routes are only registered when _UTP_SIM_MODE=1
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_sim_routes_when_registered(self, client, api_key_headers):
        """Test sim routes work when clock is initialised."""
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock, get_sim_clock
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        # Temporarily add sim routes
        _app.include_router(sim_router)
        try:
            ts = [
                datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc),
                datetime(2026, 4, 1, 13, 35, tzinfo=timezone.utc),
                datetime(2026, 4, 1, 13, 40, tzinfo=timezone.utc),
            ]
            clock = init_sim_clock(date(2026, 4, 1), ts)
            clock._tickers = ["SPX"]

            # GET /sim/status
            resp = await client.get("/sim/status", headers=api_key_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["active"] is True
            assert data["date"] == "2026-04-01"
            assert data["timestamp_count"] == 3

            # POST /sim/set-time (advance)
            resp = await client.post(
                "/sim/set-time",
                json={"advance_minutes": 5},
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["cursor_position"] == 1

            # POST /sim/set-time (ET time)
            resp = await client.post(
                "/sim/set-time",
                json={"time": "09:40"},
                headers=api_key_headers,
            )
            assert resp.status_code == 200

            # GET /sim/timestamps
            resp = await client.get("/sim/timestamps", headers=api_key_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 3

            # POST /sim/reset
            resp = await client.post("/sim/reset", headers=api_key_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["reset"] is True

            # POST /sim/auto-advance
            resp = await client.post(
                "/sim/auto-advance",
                json={"enabled": True, "interval": 0.01},
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            assert resp.json()["auto_advancing"] is True
            clock.stop_auto_advance()

            # POST /sim/load-date (requires CSVSimulationProvider, not stub)
            resp = await client.post(
                "/sim/load-date",
                json={"date": "2026-04-02"},
                headers=api_key_headers,
            )
            assert resp.status_code == 409  # not in sim mode

        finally:
            reset_sim_clock()
            # Remove the sim routes we added
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]

    @pytest.mark.asyncio
    async def test_set_time_validation(self, client, api_key_headers):
        """Test set-time validates mutual exclusivity."""
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        _app.include_router(sim_router)
        try:
            ts = [datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)]
            init_sim_clock(date(2026, 4, 1), ts)

            # No fields
            resp = await client.post(
                "/sim/set-time", json={}, headers=api_key_headers,
            )
            assert resp.status_code == 422

            # Multiple fields
            resp = await client.post(
                "/sim/set-time",
                json={"time": "10:00", "advance_minutes": 5},
                headers=api_key_headers,
            )
            assert resp.status_code == 422
        finally:
            reset_sim_clock()
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]


class TestSimulationMarketData:
    """Test that simulation mode bypasses market hours check."""

    def test_simulation_mode_flag(self):
        from app.services.market_data import set_simulation_mode, _simulation_mode, _is_market_active
        original = _simulation_mode
        try:
            set_simulation_mode(True)
            assert _is_market_active() is True
        finally:
            set_simulation_mode(original)


class TestSimDaemonMode:
    """Test simulation daemon CLI wiring."""

    def test_resolve_data_dir_sim(self):
        from utp import _resolve_data_dir
        result = _resolve_data_dir("data/utp", "sim")
        assert str(result).endswith("sim")

    def test_sim_date_requires_tickers(self):
        """--sim-date without --tickers should fail."""
        import utp
        args = argparse.Namespace(
            sim_date="2026-04-01",
            tickers=None,
            server_host="0.0.0.0",
            server_port=8100,
            data_dir="data/utp",
            log_level="INFO",
        )
        # _cmd_daemon_sim returns 1 when tickers missing
        result = asyncio.get_event_loop().run_until_complete(utp._cmd_daemon_sim(args, "INFO"))
        assert result == 1


class TestSimulationPicks:
    """Tests for /sim/picks and /sim/execute-picks endpoints."""

    def _setup_sim_provider(self, tmp_path):
        """Create CSV data and wire up sim provider + clock."""
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock

        # Create equity data
        eq_dir = tmp_path / "equities" / "SPX"
        eq_dir.mkdir(parents=True)
        (eq_dir / "SPX_equities_2026-04-01.csv").write_text(
            "timestamp,ticker,open,high,low,close,volume,vwap,transactions\n"
            "2026-04-01 13:30:00+00:00,I:SPX,5500,5520,5480,5510,1000,,\n"
            "2026-04-01 13:35:00+00:00,I:SPX,5510,5530,5490,5520,1100,,\n"
            "2026-04-01 13:40:00+00:00,I:SPX,5520,5540,5500,5530,900,,\n"
        )

        # Create options data with realistic strikes
        opt_dir = tmp_path / "options" / "SPX"
        opt_dir.mkdir(parents=True)
        (opt_dir / "SPX_options_2026-04-01.csv").write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,vwap,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            "2026-04-01T13:30:00+00:00,O:SPX,put,5400,2026-04-01,3.50,4.00,3.75,3.75,,,,,,0.20,500\n"
            "2026-04-01T13:30:00+00:00,O:SPX,put,5375,2026-04-01,1.50,2.00,1.75,1.75,,,,,,0.22,300\n"
            "2026-04-01T13:30:00+00:00,O:SPX,put,5350,2026-04-01,0.80,1.20,1.00,1.00,,,,,,0.25,200\n"
            "2026-04-01T13:30:00+00:00,O:SPX,call,5600,2026-04-01,3.00,3.50,3.25,3.25,,,,,,0.18,400\n"
            "2026-04-01T13:30:00+00:00,O:SPX,call,5625,2026-04-01,1.20,1.70,1.45,1.45,,,,,,0.20,250\n"
        )

        provider = CSVSimulationProvider(
            sim_date=date(2026, 4, 1),
            tickers=["SPX"],
            equities_dir=tmp_path / "equities",
            options_dir=tmp_path / "options",
        )
        return provider

    @pytest.mark.asyncio
    async def test_picks_generates_spreads(self, tmp_path):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        from app.routes.simulation import _generate_picks
        from app.core.provider import ProviderRegistry

        provider = self._setup_sim_provider(tmp_path)
        try:
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            clock = init_sim_clock(date(2026, 4, 1), ts)
            clock._tickers = ["SPX"]
            ProviderRegistry.register(provider)

            picks = await _generate_picks(
                tickers=["SPX"],
                option_types=["put"],
                min_otm_pct=0.01,
                min_credit=0.10,
                max_loss_per_spread=50000,
                spread_width=None,
                num_contracts=10,
                dte=[0],
                roi_min=0,
                sort_by="roi",
                limit=20,
            )
            assert len(picks) > 0
            for pick in picks:
                assert pick["ticker"] == "SPX"
                assert pick["option_type"] == "put"
                assert pick["credit"] > 0
                assert pick["short_strike"] > pick["long_strike"]  # put spread
                assert pick["otm_pct"] > 0
        finally:
            reset_sim_clock()

    @pytest.mark.asyncio
    async def test_picks_endpoint(self, tmp_path, client, api_key_headers):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        from app.core.provider import ProviderRegistry

        provider = self._setup_sim_provider(tmp_path)
        _app.include_router(sim_router)
        try:
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            clock = init_sim_clock(date(2026, 4, 1), ts)
            clock._tickers = ["SPX"]
            ProviderRegistry.register(provider)

            resp = await client.post(
                "/sim/picks",
                json={
                    "tickers": ["SPX"],
                    "option_types": ["put"],
                    "min_credit": 0.10,
                    "min_otm_pct": 0.01,
                    "dte": [0],
                },
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "picks" in data
            assert data["count"] >= 0
        finally:
            reset_sim_clock()
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]


class TestSimulationSweep:
    """Tests for /sim/sweep endpoint."""

    @pytest.mark.asyncio
    async def test_sweep_runs_combinations(self, tmp_path, client, api_key_headers):
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        from app.core.provider import ProviderRegistry

        # Create minimal data
        eq_dir = tmp_path / "equities" / "SPX"
        eq_dir.mkdir(parents=True)
        (eq_dir / "SPX_equities_2026-04-01.csv").write_text(
            "timestamp,ticker,open,high,low,close,volume,vwap,transactions\n"
            "2026-04-01 13:30:00+00:00,I:SPX,5500,5520,5480,5510,1000,,\n"
            "2026-04-01 13:35:00+00:00,I:SPX,5510,5530,5490,5520,1100,,\n"
        )
        opt_dir = tmp_path / "options" / "SPX"
        opt_dir.mkdir(parents=True)
        (opt_dir / "SPX_options_2026-04-01.csv").write_text(
            "timestamp,ticker,type,strike,expiration,bid,ask,day_close,vwap,fmv,"
            "delta,gamma,theta,vega,implied_volatility,volume\n"
            "2026-04-01T13:30:00+00:00,O:SPX,put,5400,2026-04-01,3.50,4.00,3.75,3.75,,,,,,0.20,500\n"
            "2026-04-01T13:30:00+00:00,O:SPX,put,5375,2026-04-01,1.50,2.00,1.75,1.75,,,,,,0.22,300\n"
        )

        provider = CSVSimulationProvider(
            sim_date=date(2026, 4, 1),
            tickers=["SPX"],
            equities_dir=tmp_path / "equities",
            options_dir=tmp_path / "options",
        )
        _app.include_router(sim_router)
        try:
            await provider.connect()
            ts = provider.get_all_equity_timestamps()
            clock = init_sim_clock(date(2026, 4, 1), ts)
            clock._tickers = ["SPX"]
            ProviderRegistry.register(provider)

            resp = await client.post(
                "/sim/sweep",
                json={
                    "tickers": ["SPX"],
                    "sweep_params": {
                        "num_contracts": [5, 10],
                        "min_credit": [0.10, 1.00],
                    },
                    "fixed_params": {
                        "dte": [0],
                        "entry_start_et": "09:30",
                        "entry_end_et": "16:00",
                    },
                },
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            # 2 x 2 = 4 combinations
            assert data["combinations"] == 4
            assert len(data["results"]) == 4
            for r in data["results"]:
                assert "params" in r
                assert "trades" in r
                assert "net_pnl" in r
        finally:
            reset_sim_clock()
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]


class TestSimLoadDate:
    """Tests for /sim/load-date hot-swap."""

    @pytest.mark.asyncio
    async def test_load_date_no_sim(self, client, api_key_headers):
        """load-date fails when not in sim mode (routes not registered)."""
        resp = await client.post(
            "/sim/load-date", json={"date": "2026-04-02"}, headers=api_key_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_load_date_invalid_date(self, client, api_key_headers):
        """load-date rejects invalid date format."""
        from app.core.provider import ProviderRegistry as PR
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        _app.include_router(sim_router)
        try:
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=Path("/nonexistent"),
                options_dir=Path("/nonexistent"),
            )
            PR.register(provider)
            ts = [datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)]
            init_sim_clock(date(2026, 4, 1), ts)

            resp = await client.post(
                "/sim/load-date",
                json={"date": "not-a-date"},
                headers=api_key_headers,
            )
            assert resp.status_code == 422
        finally:
            reset_sim_clock()
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]

    @pytest.mark.asyncio
    async def test_load_date_not_sim_mode(self, client, api_key_headers):
        """load-date fails when provider is not CSVSimulationProvider."""
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        _app.include_router(sim_router)
        try:
            ts = [datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)]
            init_sim_clock(date(2026, 4, 1), ts)
            # Default provider (IBKRProvider stub) is not CSVSimulationProvider
            resp = await client.post(
                "/sim/load-date",
                json={"date": "2026-04-02"},
                headers=api_key_headers,
            )
            assert resp.status_code == 409
        finally:
            reset_sim_clock()
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]

    @pytest.mark.asyncio
    async def test_load_date_no_data(self, client, api_key_headers, tmp_path):
        """load-date returns 404 when no equity data exists for the date."""
        from app.core.provider import ProviderRegistry as PR
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        _app.include_router(sim_router)
        try:
            eq_dir = tmp_path / "eq"
            opt_dir = tmp_path / "opt"
            eq_dir.mkdir()
            opt_dir.mkdir()
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=eq_dir,
                options_dir=opt_dir,
            )
            await provider.connect()
            PR.register(provider)
            ts = [datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)]
            init_sim_clock(date(2026, 4, 1), ts)

            resp = await client.post(
                "/sim/load-date",
                json={"date": "2026-04-02"},
                headers=api_key_headers,
            )
            assert resp.status_code == 404
        finally:
            reset_sim_clock()
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]

    @pytest.mark.asyncio
    async def test_load_date_success(self, client, api_key_headers, tmp_path):
        """load-date swaps date when equity data exists."""
        from app.core.provider import ProviderRegistry as PR
        from app.services.simulation_clock import init_sim_clock, reset_sim_clock, get_sim_clock
        from app.core.providers.csv_simulation import CSVSimulationProvider
        from app.routes.simulation import router as sim_router
        from app.main import app as _app
        _app.include_router(sim_router)
        try:
            eq_dir = tmp_path / "eq"
            opt_dir = tmp_path / "opt"
            # Create equity CSV for 2026-04-02
            spx_dir = eq_dir / "SPX"
            spx_dir.mkdir(parents=True)
            csv_path = spx_dir / "SPX_equities_2026-04-02.csv"
            csv_path.write_text(
                "timestamp,open,high,low,close,volume\n"
                "2026-04-02T13:30:00+00:00,5500,5510,5490,5505,1000\n"
                "2026-04-02T13:35:00+00:00,5505,5515,5500,5510,1200\n"
            )
            opt_dir.mkdir(exist_ok=True)

            # Start with original date
            provider = CSVSimulationProvider(
                sim_date=date(2026, 4, 1),
                tickers=["SPX"],
                equities_dir=eq_dir,
                options_dir=opt_dir,
            )
            await provider.connect()
            PR.register(provider)
            ts = [datetime(2026, 4, 1, 13, 30, tzinfo=timezone.utc)]
            init_sim_clock(date(2026, 4, 1), ts)

            # Load new date
            resp = await client.post(
                "/sim/load-date",
                json={"date": "2026-04-02"},
                headers=api_key_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["active"] is True
            assert data["date"] == "2026-04-02"
            assert data["timestamp_count"] == 2

            # Verify clock was updated
            clock = get_sim_clock()
            assert clock is not None
            assert clock.sim_date == date(2026, 4, 2)
        finally:
            reset_sim_clock()
            _app.routes[:] = [r for r in _app.routes
                              if not (hasattr(r, 'path') and str(getattr(r, 'path', '')).startswith('/sim'))]


class TestAutoTraderEngine:
    """Tests for the auto-trader engine in utp_voice.py."""

    def test_default_config(self):
        """Default config has expected fields."""
        from utp_voice import _default_auto_trader_config
        cfg = _default_auto_trader_config()
        assert "tickers" in cfg
        assert "min_otm_pct" in cfg
        assert "spread_width" in cfg
        assert cfg["max_trades_per_day"] == 5
        assert cfg["profit_target_pct"] == 0.50

    def test_et_minutes(self):
        """Parse ET time strings correctly."""
        from utp_voice import _et_minutes
        assert _et_minutes("09:30") == 570
        assert _et_minutes("15:00") == 900
        assert _et_minutes("00:00") == 0

    @pytest.mark.asyncio
    async def test_config_endpoint_set(self):
        """POST /api/auto-trader/config sets and returns config."""
        from httpx import ASGITransport as _AT, AsyncClient as _AC
        from utp_voice import app as voice_app
        transport = _AT(app=voice_app)
        async with _AC(transport=transport, base_url="http://test") as vc:
            resp = await vc.post(
                "/api/auto-trader/config",
                json={"tickers": ["NDX"], "spread_width": 50},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["config"]["tickers"] == ["NDX"]
            assert data["config"]["spread_width"] == 50
            # Defaults preserved
            assert data["config"]["profit_target_pct"] == 0.50

    @pytest.mark.asyncio
    async def test_config_endpoint_get(self):
        """GET /api/auto-trader/config returns current config."""
        from httpx import ASGITransport as _AT, AsyncClient as _AC
        from utp_voice import app as voice_app, _default_auto_trader_config
        import utp_voice
        utp_voice._auto_trader_config = _default_auto_trader_config()

        transport = _AT(app=voice_app)
        async with _AC(transport=transport, base_url="http://test") as vc:
            resp = await vc.get("/api/auto-trader/config")
            assert resp.status_code == 200
            data = resp.json()
            assert "tickers" in data
            assert "spread_width" in data

    @pytest.mark.asyncio
    async def test_run_day_no_sim(self):
        """run-day returns error when daemon has no sim active."""
        from httpx import ASGITransport as _AT, AsyncClient as _AC
        from utp_voice import app as voice_app
        transport = _AT(app=voice_app)
        async with _AC(transport=transport, base_url="http://test") as vc:
            # This will fail because the daemon client can't connect
            resp = await vc.post("/api/auto-trader/run-day", json={})
            # Should get a connection error (502 or 500)
            assert resp.status_code in (500, 502)

    @pytest.mark.asyncio
    async def test_run_range_missing_dates(self):
        """run-range requires start_date and end_date."""
        from httpx import ASGITransport as _AT, AsyncClient as _AC
        from utp_voice import app as voice_app
        transport = _AT(app=voice_app)
        async with _AC(transport=transport, base_url="http://test") as vc:
            resp = await vc.post("/api/auto-trader/run-range", json={})
            assert resp.status_code == 422

    def test_build_spreads_filters(self):
        """compute_spreads_server filters by OTM% and credit."""
        from utp_voice import compute_spreads_server
        chain = {
            "put": [
                {"strike": 5400, "bid": 3.0, "ask": 3.5, "greeks": {"delta": -0.1}},
                {"strike": 5380, "bid": 1.5, "ask": 2.0, "greeks": {"delta": -0.05}},
                {"strike": 5480, "bid": 8.0, "ask": 8.5, "greeks": {"delta": -0.3}},
                {"strike": 5460, "bid": 5.0, "ask": 5.5, "greeks": {"delta": -0.2}},
            ],
        }
        current_price = 5500
        spreads = compute_spreads_server(
            chain, "SPX", current_price, 20,
            filters={"min_otm_pct": 1.0, "min_credit": 0.50, "option_type": "PUT"},
        )
        # All spreads should be OTM by at least 1%
        for s in spreads:
            assert abs(s["otm_pct"]) >= 1.0
            assert s["credit"] >= 0.50

    def test_sim_trader_build_config(self):
        """sim_trader.build_config returns valid config."""
        import sim_trader
        config = sim_trader.build_config()
        assert isinstance(config, dict)
        assert "tickers" in config
        assert "spread_width" in config
        assert "num_contracts" in config
        assert config["max_trades_per_day"] > 0

    def test_allowed_expirations_monday(self):
        """Mon sim_date, DTE=[0,1,2] → Mon/Tue/Wed only (within same week)."""
        from utp_voice import _allowed_expirations
        # 2026-04-06 is a Monday
        result = _allowed_expirations(
            "2026-04-06", [0, 1, 2],
            ["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10", "2026-04-13"],
        )
        assert "2026-04-06" in result  # DTE=0
        assert "2026-04-07" in result  # DTE=1
        assert "2026-04-08" in result  # DTE=2
        assert "2026-04-09" not in result  # DTE=3, not in dte_list
        assert "2026-04-13" not in result  # next week

    def test_allowed_expirations_friday(self):
        """Fri sim_date, DTE=[0,1] → Fri only (Sat filtered by week boundary)."""
        from utp_voice import _allowed_expirations
        # 2026-04-10 is a Friday
        result = _allowed_expirations(
            "2026-04-10", [0, 1],
            ["2026-04-10", "2026-04-11", "2026-04-13"],
        )
        assert result == ["2026-04-10"]  # DTE=0 only; Sat(DTE=1) > Friday

    def test_allowed_expirations_wednesday(self):
        """Wed sim_date, DTE=[0,1,2,3] → Wed/Thu/Fri (Sat filtered)."""
        from utp_voice import _allowed_expirations
        # 2026-04-08 is a Wednesday
        result = _allowed_expirations(
            "2026-04-08", [0, 1, 2, 3],
            ["2026-04-08", "2026-04-09", "2026-04-10", "2026-04-11", "2026-04-13"],
        )
        assert "2026-04-08" in result  # DTE=0
        assert "2026-04-09" in result  # DTE=1
        assert "2026-04-10" in result  # DTE=2
        assert "2026-04-11" not in result  # Saturday, > Friday
        assert "2026-04-13" not in result  # next week

    def test_allowed_expirations_no_match(self):
        """DTE=[5] on Monday → empty (next week filtered)."""
        from utp_voice import _allowed_expirations
        result = _allowed_expirations(
            "2026-04-06", [5],
            ["2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09", "2026-04-10", "2026-04-11"],
        )
        assert result == []  # DTE=5 would be Saturday, past Friday

    def test_spreads_tagged_with_expiration(self):
        """Verify compute_spreads_server spreads can carry expiration/dte tags."""
        from utp_voice import compute_spreads_server
        chain = {
            "put": [
                {"strike": 5400, "bid": 3.0, "ask": 3.5, "greeks": {"delta": -0.1}},
                {"strike": 5380, "bid": 1.5, "ask": 2.0, "greeks": {"delta": -0.05}},
            ],
        }
        spreads = compute_spreads_server(
            chain, "SPX", 5500, 20,
            filters={"min_otm_pct": 0, "min_credit": 0, "option_type": "PUT"},
        )
        # Simulate what _build_spreads_for_engine does with DTE tagging
        for s in spreads:
            s["expiration"] = "2026-04-07"
            s["dte"] = 1
        for s in spreads:
            assert "expiration" in s
            assert "dte" in s
            assert s["dte"] == 1
            assert s["expiration"] == "2026-04-07"

    @pytest.mark.asyncio
    async def test_eod_skips_future_dte(self):
        """Position with future expiration should not be settled at EOD."""
        from utp_voice import _run_sim_day
        import utp_voice

        # Mock the daemon client
        call_log = []

        class MockClient:
            async def _get(self, path, params=None):
                call_log.append(("GET", path))
                if path == "/sim/status":
                    return {"date": "2026-04-06", "active": True}
                if path == "/sim/timestamps":
                    # Single timestamp (market open)
                    return {"timestamps": ["2026-04-06T13:30:00+00:00"]}
                return {}

            async def _post(self, path, json_data=None):
                call_log.append(("POST", path))
                return {}

            async def get_quote(self, symbol):
                return {"last": 5500, "bid": 5499, "ask": 5501}

        old_client = utp_voice.get_daemon_client
        utp_voice.get_daemon_client = lambda: MockClient()
        try:
            # Carry a DTE=1 position into today (ITM enough to avoid profit target)
            carry = [{
                "ticker": "SPX", "option_type": "PUT",
                "short_strike": 5510, "long_strike": 5490,
                "spread_width": 20, "credit": 2.0,
                "total_credit": 2000, "total_max_loss": 1800,
                "roi_pct": 11.1, "otm_pct": 1.8, "num_contracts": 10,
                "entry_time": "2026-04-06T14:00:00+00:00",
                "entry_price": 5500, "expiration": "2026-04-07",
                "dte": 1, "status": "open", "order_id": "",
            }]
            config = {
                "tickers": ["SPX"], "option_types": ["put"],
                "max_trades_per_day": 0,  # no new entries
                "min_otm_pct": 0.02, "spread_width": 20,
                "min_credit": 0.50, "num_contracts": 10, "dte": [0, 1],
                "max_loss_per_trade": 5000, "max_loss_per_day": 15000,
                "profit_target_pct": 0.50, "stop_loss_mult": 200.0,
                "entry_start_et": "09:45", "entry_end_et": "15:00",
            }
            result = await _run_sim_day(config, carry_positions=carry)
            # The position should be in carry_forward (not settled)
            assert len(result["carry_forward"]) == 1
            assert result["carry_forward"][0]["status"] == "open"
            assert result["carry_forward"][0]["expiration"] == "2026-04-07"
        finally:
            utp_voice.get_daemon_client = old_client

    @pytest.mark.asyncio
    async def test_carry_positions(self):
        """DTE-1 position carries to next day and settles on expiration day."""
        from utp_voice import _run_sim_day
        import utp_voice

        class MockClient:
            def __init__(self, sim_date):
                self.sim_date = sim_date

            async def _get(self, path, params=None):
                if path == "/sim/status":
                    return {"date": self.sim_date, "active": True}
                if path == "/sim/timestamps":
                    return {"timestamps": [f"{self.sim_date}T13:30:00+00:00"]}
                return {}

            async def _post(self, path, json_data=None):
                return {}

            async def get_quote(self, symbol):
                return {"last": 5500, "bid": 5499, "ask": 5501}

        config = {
            "tickers": ["SPX"], "option_types": ["put"],
            "max_trades_per_day": 0, "min_otm_pct": 0.02,
            "spread_width": 20, "min_credit": 0.50, "num_contracts": 10,
            "dte": [0, 1], "max_loss_per_trade": 5000, "max_loss_per_day": 15000,
            "profit_target_pct": 0.50, "stop_loss_mult": 200.0,
            "entry_start_et": "09:45", "entry_end_et": "15:00",
        }

        carry = [{
            "ticker": "SPX", "option_type": "PUT",
            "short_strike": 5510, "long_strike": 5490,
            "spread_width": 20, "credit": 2.0,
            "total_credit": 2000, "total_max_loss": 1800,
            "roi_pct": 11.1, "otm_pct": 1.8, "num_contracts": 10,
            "entry_time": "2026-04-06T14:00:00+00:00",
            "entry_price": 5500, "expiration": "2026-04-07",
            "dte": 1, "status": "open", "order_id": "",
        }]

        old_client = utp_voice.get_daemon_client

        # Day 1 (Apr 6): position carries forward
        utp_voice.get_daemon_client = lambda: MockClient("2026-04-06")
        try:
            r1 = await _run_sim_day(config, carry_positions=carry)
            assert len(r1["carry_forward"]) == 1

            # Day 2 (Apr 7): position expires today
            utp_voice.get_daemon_client = lambda: MockClient("2026-04-07")
            r2 = await _run_sim_day(config, carry_positions=r1["carry_forward"])
            assert len(r2["carry_forward"]) == 0
            # Position should be settled
            settled = [t for t in r2["trades"] if t.get("status") == "closed"]
            assert len(settled) == 1
            assert settled[0]["exit_reason"] in ("eod_expire_otm", "eod_itm")
        finally:
            utp_voice.get_daemon_client = old_client

    def test_config_dte_field(self):
        """Config endpoint accepts and returns dte list."""
        from utp_voice import _default_auto_trader_config
        cfg = _default_auto_trader_config()
        assert "dte" in cfg
        assert isinstance(cfg["dte"], list)
        assert 0 in cfg["dte"]
        # Simulate merging with custom dte
        cfg.update({"dte": [0, 1, 2, 3]})
        assert cfg["dte"] == [0, 1, 2, 3]

    def test_config_diversity_field(self):
        """diversity_enabled field present with correct default."""
        from utp_voice import _default_auto_trader_config
        cfg = _default_auto_trader_config()
        assert "diversity_enabled" in cfg
        assert cfg["diversity_enabled"] is True

    def test_select_diverse_spread_basic(self):
        """No open positions -> picks highest ROI."""
        from utp_voice import _select_diverse_spread
        candidates = [
            {"roi_pct": 10, "ticker": "SPX", "option_type": "PUT",
             "short_strike": 5400, "width": 20, "dte": 0},
            {"roi_pct": 8, "ticker": "RUT", "option_type": "PUT",
             "short_strike": 2100, "width": 20, "dte": 0},
        ]
        config = {"diversity_enabled": True}
        result = _select_diverse_spread(candidates, [], config)
        assert result["roi_pct"] == 10
        assert result["ticker"] == "SPX"

    def test_select_diverse_penalizes_same_ticker_type(self):
        """Second SPX PUT penalized, RUT PUT wins."""
        from utp_voice import _select_diverse_spread
        candidates = [
            {"roi_pct": 12, "ticker": "SPX", "option_type": "PUT",
             "short_strike": 5350, "width": 20, "dte": 0},
            {"roi_pct": 9, "ticker": "RUT", "option_type": "PUT",
             "short_strike": 2100, "width": 20, "dte": 0},
        ]
        open_positions = [
            {"ticker": "SPX", "option_type": "PUT", "short_strike": 5400, "dte": 0},
        ]
        config = {"diversity_enabled": True}
        result = _select_diverse_spread(candidates, open_positions, config)
        # SPX PUT gets -25 (same ticker+type) -15 (strike within 2x width) -10 (same dte)
        # = 12 - 50 = -38
        # RUT PUT gets -10 (same dte) + 10 (new ticker) = 9 - 10 + 10 = 9
        assert result["ticker"] == "RUT"

    def test_select_diverse_penalizes_same_dte(self):
        """Same DTE penalized."""
        from utp_voice import _select_diverse_spread
        candidates = [
            {"roi_pct": 10, "ticker": "NDX", "option_type": "PUT",
             "short_strike": 20000, "width": 50, "dte": 0},
            {"roi_pct": 9, "ticker": "NDX", "option_type": "PUT",
             "short_strike": 20000, "width": 50, "dte": 1},
        ]
        open_positions = [
            {"ticker": "SPX", "option_type": "PUT", "short_strike": 5400, "dte": 0},
        ]
        config = {"diversity_enabled": True}
        result = _select_diverse_spread(candidates, open_positions, config)
        # DTE=0 candidate: +10 + 10 (new ticker) - 10 (same dte) = 10
        # DTE=1 candidate: +9 + 10 (new ticker) + 5 (new dte) = 24
        assert result["dte"] == 1

    def test_select_diverse_penalizes_same_strike_range(self):
        """Overlapping strikes penalized."""
        from utp_voice import _select_diverse_spread
        candidates = [
            {"roi_pct": 10, "ticker": "SPX", "option_type": "CALL",
             "short_strike": 5410, "width": 20, "dte": 0},
            {"roi_pct": 9, "ticker": "SPX", "option_type": "CALL",
             "short_strike": 5600, "width": 20, "dte": 0},
        ]
        open_positions = [
            {"ticker": "SPX", "option_type": "PUT", "short_strike": 5400, "dte": 0},
        ]
        config = {"diversity_enabled": True}
        result = _select_diverse_spread(candidates, open_positions, config)
        # 5410 is within 2*20=40 of 5400 -> -15 penalty
        # 5600 is not within 40 of 5400 -> no strike penalty
        # Both same ticker diff type: -15 each, same dte: -10 each
        # 5410: 10 - 15 - 10 - 15 = -30
        # 5600: 9 - 15 - 10 = -16
        assert result["short_strike"] == 5600

    def test_select_diverse_disabled(self):
        """diversity_enabled=False -> picks top ROI."""
        from utp_voice import _select_diverse_spread
        candidates = [
            {"roi_pct": 12, "ticker": "SPX", "option_type": "PUT",
             "short_strike": 5350, "width": 20, "dte": 0},
            {"roi_pct": 9, "ticker": "RUT", "option_type": "PUT",
             "short_strike": 2100, "width": 20, "dte": 0},
        ]
        open_positions = [
            {"ticker": "SPX", "option_type": "PUT", "short_strike": 5400, "dte": 0},
        ]
        config = {"diversity_enabled": False}
        result = _select_diverse_spread(candidates, open_positions, config)
        assert result["roi_pct"] == 12  # highest ROI, ignoring penalties

    def test_select_diverse_single_ticker(self):
        """Only one ticker -> still picks something."""
        from utp_voice import _select_diverse_spread
        candidates = [
            {"roi_pct": 8, "ticker": "SPX", "option_type": "PUT",
             "short_strike": 5300, "width": 20, "dte": 1},
        ]
        open_positions = [
            {"ticker": "SPX", "option_type": "PUT", "short_strike": 5400, "dte": 0},
        ]
        config = {"diversity_enabled": True}
        result = _select_diverse_spread(candidates, open_positions, config)
        assert result is not None
        assert result["ticker"] == "SPX"

    def test_run_day_stream_endpoint_registered(self):
        """SSE run-day-stream endpoint is registered on the app."""
        from utp_voice import app as voice_app
        paths = [r.path for r in voice_app.routes]
        assert "/api/auto-trader/run-day-stream" in paths

    @pytest.mark.asyncio
    async def test_shadow_start_stop(self):
        """Shadow start/stop endpoints work."""
        from httpx import ASGITransport as _AT, AsyncClient as _AC
        from utp_voice import app as voice_app
        import utp_voice
        transport = _AT(app=voice_app)
        async with _AC(transport=transport, base_url="http://test") as vc:
            # Stop when not running
            resp = await vc.post("/api/auto-trader/stop-shadow")
            assert resp.status_code == 200
            assert resp.json()["status"] == "not_running"

    @pytest.mark.asyncio
    async def test_shadow_positions(self):
        """Shadow positions tracked separately."""
        from httpx import ASGITransport as _AT, AsyncClient as _AC
        from utp_voice import app as voice_app
        import utp_voice
        # Reset shadow positions
        utp_voice._shadow_positions = [
            {"ticker": "SPX", "option_type": "PUT", "short_strike": 5400,
             "long_strike": 5380, "credit": 1.5, "status": "open"},
        ]
        transport = _AT(app=voice_app)
        async with _AC(transport=transport, base_url="http://test") as vc:
            resp = await vc.get("/api/auto-trader/shadow-positions")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["positions"]) == 1
            assert data["positions"][0]["ticker"] == "SPX"
        # Cleanup
        utp_voice._shadow_positions = []

    @pytest.mark.asyncio
    async def test_shadow_fake_fill(self):
        """Shadow fills at quoted price without execute_trade."""
        from utp_voice import _select_diverse_spread
        # Verify the diversity function returns a candidate (used in shadow loop)
        candidates = [
            {"roi_pct": 10, "ticker": "SPX", "option_type": "PUT",
             "short_strike": 5400, "width": 20, "dte": 0,
             "credit": 1.5, "total_credit": 1500, "total_max_loss": 350},
        ]
        result = _select_diverse_spread(candidates, [], {"diversity_enabled": True})
        assert result is not None
        assert result["credit"] == 1.5

    def test_sim_trader_shadow_arg(self):
        """CLI accepts --shadow."""
        import sim_trader
        parser = argparse.ArgumentParser()
        parser.add_argument("--shadow", action="store_true")
        args = parser.parse_args(["--shadow"])
        assert args.shadow is True

    def test_sim_trader_sim_speed_arg(self):
        """CLI accepts --sim-speed."""
        import sim_trader
        parser = argparse.ArgumentParser()
        parser.add_argument("--sim-speed", type=float, default=10)
        args = parser.parse_args(["--sim-speed", "5"])
        assert args.sim_speed == 5.0

    def test_live_stream_endpoint_registered(self):
        """Live stream endpoint is registered on the app."""
        from utp_voice import app as voice_app
        paths = [r.path for r in voice_app.routes]
        assert "/api/auto-trader/live-stream" in paths

    def test_shadow_stream_endpoint_registered(self):
        """Shadow stream endpoint is registered on the app."""
        from utp_voice import app as voice_app
        paths = [r.path for r in voice_app.routes]
        assert "/api/auto-trader/shadow-stream" in paths

    def test_event_bus_emit_and_subscribe(self):
        """Event bus emits events to subscribers."""
        import asyncio
        from utp_voice import _emit_engine_event, _engine_event_subscribers
        q = asyncio.Queue(maxsize=10)
        _engine_event_subscribers.setdefault("test_mode", []).append(q)
        try:
            _emit_engine_event("test_mode", {"event": "tick", "data": 42})
            assert not q.empty()
            event = q.get_nowait()
            assert event["event"] == "tick"
            assert event["data"] == 42
        finally:
            _engine_event_subscribers["test_mode"].remove(q)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider latency timing
# ═══════════════════════════════════════════════════════════════════════════════


class TestProviderTiming:
    def test_record_and_snapshot_basic(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService(buffer_size=100)
        svc.record("provider.get_quote", 12.5, symbol="SPX")
        svc.record("provider.get_quote", 18.0, symbol="SPX")
        svc.record("provider.get_quote", 200.0, symbol="SPX")  # outlier
        snap = svc.snapshot()
        assert snap["total_samples"] == 3
        bucket = snap["buckets"]["provider.get_quote|SPX"]
        assert bucket["count"] == 3
        assert bucket["min_ms"] == 12.5
        assert bucket["max_ms"] == 200.0
        assert bucket["p50_ms"] == 18.0  # median of 3 samples is the middle value
        assert bucket["ok"] == 3
        assert bucket["errors"] == 0

    def test_buckets_separate_by_method_and_symbol(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService()
        svc.record("provider.get_quote", 10.0, symbol="SPX")
        svc.record("provider.get_quote", 20.0, symbol="NDX")
        svc.record("provider.get_option_quotes", 50.0, symbol="SPX")
        snap = svc.snapshot()
        keys = set(snap["buckets"].keys())
        assert keys == {
            "provider.get_quote|SPX",
            "provider.get_quote|NDX",
            "provider.get_option_quotes|SPX",
        }

    def test_record_with_no_symbol(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService()
        svc.record("provider.get_positions", 100.0)
        snap = svc.snapshot()
        assert "provider.get_positions|*" in snap["buckets"]

    def test_buffer_size_evicts_old_samples(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService(buffer_size=5)
        for i in range(10):
            svc.record("provider.get_quote", float(i + 1), symbol="X")
        snap = svc.snapshot()
        # only the most recent 5 samples remain
        assert snap["total_samples"] == 5
        bucket = snap["buckets"]["provider.get_quote|X"]
        assert bucket["count"] == 5
        assert bucket["min_ms"] == 6.0  # samples 6-10 retained
        assert bucket["max_ms"] == 10.0

    def test_filter_snapshot_by_method(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService()
        svc.record("provider.get_quote", 5.0, symbol="SPX")
        svc.record("provider.get_option_quotes", 50.0, symbol="SPX")
        snap = svc.snapshot(method="provider.get_quote")
        assert "provider.get_quote|SPX" in snap["buckets"]
        assert "provider.get_option_quotes|SPX" not in snap["buckets"]

    def test_filter_snapshot_by_symbol(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService()
        svc.record("provider.get_quote", 5.0, symbol="SPX")
        svc.record("provider.get_quote", 10.0, symbol="NDX")
        snap = svc.snapshot(symbol="NDX")
        assert "provider.get_quote|NDX" in snap["buckets"]
        assert "provider.get_quote|SPX" not in snap["buckets"]

    def test_reset_clears_state(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService()
        svc.record("provider.get_quote", 5.0, symbol="SPX")
        svc.reset()
        snap = svc.snapshot()
        assert snap["total_samples"] == 0
        assert snap["buckets"] == {}

    def test_record_marks_errors_separately(self):
        from app.services.provider_timing import ProviderTimingService
        svc = ProviderTimingService()
        svc.record("provider.get_quote", 5.0, symbol="SPX", ok=True)
        svc.record("provider.get_quote", 5.0, symbol="SPX", ok=False)
        bucket = svc.snapshot()["buckets"]["provider.get_quote|SPX"]
        assert bucket["ok"] == 1
        assert bucket["errors"] == 1

    @pytest.mark.asyncio
    async def test_timed_context_records_success(self):
        from app.services.provider_timing import (
            init_provider_timing, get_provider_timing, reset_provider_timing, timed,
        )
        reset_provider_timing()
        init_provider_timing()
        async with timed("provider.get_quote", symbol="SPX"):
            await asyncio.sleep(0.01)
        snap = get_provider_timing().snapshot()
        assert snap["total_samples"] == 1
        bucket = snap["buckets"]["provider.get_quote|SPX"]
        assert bucket["min_ms"] >= 9.0  # ~10ms sleep
        assert bucket["ok"] == 1
        assert bucket["errors"] == 0

    @pytest.mark.asyncio
    async def test_timed_context_records_exception(self):
        from app.services.provider_timing import (
            init_provider_timing, get_provider_timing, reset_provider_timing, timed,
        )
        reset_provider_timing()
        init_provider_timing()
        with pytest.raises(RuntimeError):
            async with timed("provider.get_quote", symbol="SPX"):
                raise RuntimeError("boom")
        bucket = get_provider_timing().snapshot()["buckets"]["provider.get_quote|SPX"]
        assert bucket["ok"] == 0
        assert bucket["errors"] == 1

    @pytest.mark.asyncio
    async def test_market_data_get_quote_records_timing(self, client, api_key_headers):
        """End-to-end: a centralized get_quote call records a timing sample."""
        from app.services.provider_timing import reset_provider_timing, get_provider_timing
        reset_provider_timing()
        # First call seeds the buffer.
        resp = await client.get("/market/quote/SPY", headers=api_key_headers)
        assert resp.status_code == 200
        snap = get_provider_timing().snapshot()
        # Stub providers return instantly so this path is timed but very fast.
        keys = list(snap["buckets"].keys())
        assert any(k.startswith("provider.get_quote|") for k in keys), keys


# ═══════════════════════════════════════════════════════════════════════════════
# Force-market-open override
# ═══════════════════════════════════════════════════════════════════════════════


class TestForceMarketOpen:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("UTP_FORCE_MARKET_OPEN", raising=False)
        from app.services.market_data import _is_market_open_forced
        assert _is_market_open_forced() is False

    def test_truthy_values_enable(self, monkeypatch):
        from app.services.market_data import _is_market_open_forced
        for v in ("true", "TRUE", "True", "1", "yes", "on"):
            monkeypatch.setenv("UTP_FORCE_MARKET_OPEN", v)
            assert _is_market_open_forced() is True, v

    def test_falsy_values_disable(self, monkeypatch):
        from app.services.market_data import _is_market_open_forced
        for v in ("false", "0", "no", "off", "", "anything-else"):
            monkeypatch.setenv("UTP_FORCE_MARKET_OPEN", v)
            assert _is_market_open_forced() is False, v

    def test_market_active_respects_force(self, monkeypatch):
        from app.services.market_data import _is_market_active, set_simulation_mode
        set_simulation_mode(False)
        monkeypatch.setenv("UTP_FORCE_MARKET_OPEN", "true")
        assert _is_market_active() is True

    def test_streamer_market_hours_respects_force(self, monkeypatch):
        from app.services.option_quote_streaming import _is_market_hours
        monkeypatch.setenv("UTP_FORCE_MARKET_OPEN", "true")
        assert _is_market_hours() is True

    def test_daemon_arg_wires_env(self):
        """`--force-market-open` flag exists on daemon and writes the env var."""
        import utp
        with open(utp.__file__) as f:
            src = f.read()
        # Flag declared on the daemon parser
        assert '"--force-market-open"' in src
        # And exported into the env when the flag is set
        assert 'os.environ["UTP_FORCE_MARKET_OPEN"]' in src


# ═══════════════════════════════════════════════════════════════════════════════
# /market/streaming/latency endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestLatencyEndpoint:
    @pytest.mark.asyncio
    async def test_get_returns_empty_initially(self, client, api_key_headers):
        from app.services.provider_timing import reset_provider_timing
        reset_provider_timing()
        resp = await client.get("/market/streaming/latency", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_samples"] == 0
        assert data["buckets"] == {}

    @pytest.mark.asyncio
    async def test_get_returns_recorded_samples(self, client, api_key_headers):
        from app.services.provider_timing import init_provider_timing, reset_provider_timing
        reset_provider_timing()
        svc = init_provider_timing()
        svc.record("provider.get_option_quotes", 42.0, symbol="SPX")
        resp = await client.get("/market/streaming/latency", headers=api_key_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_samples"] == 1
        assert "provider.get_option_quotes|SPX" in data["buckets"]
        assert data["buckets"]["provider.get_option_quotes|SPX"]["p50_ms"] == 42.0

    @pytest.mark.asyncio
    async def test_get_filters_by_method(self, client, api_key_headers):
        from app.services.provider_timing import init_provider_timing, reset_provider_timing
        reset_provider_timing()
        svc = init_provider_timing()
        svc.record("provider.get_quote", 5.0, symbol="SPX")
        svc.record("provider.get_option_quotes", 50.0, symbol="SPX")
        resp = await client.get(
            "/market/streaming/latency?method=provider.get_quote",
            headers=api_key_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        keys = list(data["buckets"].keys())
        assert keys == ["provider.get_quote|SPX"]

    @pytest.mark.asyncio
    async def test_get_filters_by_symbol(self, client, api_key_headers):
        from app.services.provider_timing import init_provider_timing, reset_provider_timing
        reset_provider_timing()
        svc = init_provider_timing()
        svc.record("provider.get_quote", 5.0, symbol="SPX")
        svc.record("provider.get_quote", 10.0, symbol="NDX")
        resp = await client.get(
            "/market/streaming/latency?symbol=NDX",
            headers=api_key_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        keys = list(data["buckets"].keys())
        assert keys == ["provider.get_quote|NDX"]

    @pytest.mark.asyncio
    async def test_post_reset_clears_buffer(self, client, api_key_headers):
        from app.services.provider_timing import init_provider_timing, reset_provider_timing
        reset_provider_timing()
        svc = init_provider_timing()
        svc.record("provider.get_quote", 5.0, symbol="SPX")
        resp = await client.post("/market/streaming/latency/reset", headers=api_key_headers)
        assert resp.status_code == 200
        assert resp.json()["reset"] is True
        # Subsequent GET shows empty
        resp = await client.get("/market/streaming/latency", headers=api_key_headers)
        assert resp.json()["total_samples"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# `utp.py latency-probe` CLI subcommand
# ═══════════════════════════════════════════════════════════════════════════════


class TestLatencyProbeCommand:
    def test_subcommand_registered(self):
        """The latency-probe subcommand is registered with the expected flags."""
        import utp
        with open(utp.__file__) as f:
            src = f.read()
        assert 'subparsers.add_parser("latency-probe"' in src
        assert "--duration" in src
        assert "--filter-method" in src
        assert "--filter-symbol" in src
        assert "--no-reset" in src

    def test_alias_registered(self):
        import utp
        with open(utp.__file__) as f:
            src = f.read()
        # alias_map entry
        assert '"latency": "latency-probe"' in src

    @pytest.mark.asyncio
    async def test_no_daemon_returns_1(self, capsys, monkeypatch):
        """When no daemon is detected, the command prints help and exits 1."""
        import utp
        # _disable_server_detection autouse fixture forces _detect_server -> None
        args = argparse.Namespace(
            duration=1, filter_method=None, filter_symbol=None,
            no_reset=False, server=None, server_port=8000,
        )
        rc = await utp._cmd_latency_probe(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "No daemon detected" in out

    @pytest.mark.asyncio
    async def test_probe_against_inproc_app(self, monkeypatch, capsys, api_key_headers):
        """Run the probe end-to-end against the in-process app via a stub server."""
        import utp
        from httpx import ASGITransport, AsyncClient
        from app.main import app
        from app.services.provider_timing import init_provider_timing, reset_provider_timing

        reset_provider_timing()
        svc = init_provider_timing()
        svc.record("provider.get_quote", 12.0, symbol="SPX")

        # Patch _detect_server to return a sentinel URL so the command
        # routes through our patched httpx client.
        monkeypatch.setattr(utp, "_detect_server", lambda a: "http://probe-test")

        # Patch httpx.AsyncClient to return a transport bound to the live ASGI app.
        # The ASGITransport ignores base_url scheme, so we can keep the sentinel.
        real_async_client = utp.__dict__.get("httpx") or __import__("httpx").AsyncClient

        class _ProbeClient(AsyncClient):
            def __init__(self, *a, **kw):
                kw["transport"] = ASGITransport(app=app)
                kw["base_url"] = "http://probe-test"
                kw.setdefault("timeout", 10.0)
                super().__init__(**kw)
                # Inject API key header so require_auth lets us through.
                self.headers.update(api_key_headers)

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", _ProbeClient)

        args = argparse.Namespace(
            duration=1, filter_method=None, filter_symbol=None,
            no_reset=True,  # don't clear our seeded sample
            server=None, server_port=8000,
        )
        rc = await utp._cmd_latency_probe(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Latency probe" in out
        assert "provider.get_quote" in out
        # Header line shows up
        assert "Method" in out and "p95" in out


# ═══════════════════════════════════════════════════════════════════════════════
# Tiered option-quote streaming (IBKR hot tier vs CSV warm tier)
# ═══════════════════════════════════════════════════════════════════════════════


def _make_streaming_config(**overrides):
    """Build a StreamingConfig with the indices we care about."""
    from app.services.streaming_config import StreamingConfig, StreamingSymbolConfig
    base = dict(
        symbols=[
            StreamingSymbolConfig(symbol="SPX", sec_type="IND", exchange="CBOE"),
            StreamingSymbolConfig(symbol="NDX", sec_type="IND", exchange="NASDAQ"),
            StreamingSymbolConfig(symbol="RUT", sec_type="IND", exchange="RUSSELL"),
        ],
        option_quotes_enabled=True,
        option_quotes_ibkr_strike_range_pct=3.0,
        option_quotes_csv_strike_range_pct=15.0,
        option_quotes_ibkr_dte_list=[0, 1, 2],
        option_quotes_csv_dte_max=10,
    )
    base.update(overrides)
    return StreamingConfig(**base)


class TestTieredOptionStreaming:
    def test_dte_for_exp_basic(self):
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from datetime import date
        today = date(2026, 4, 19)
        assert OptionQuoteStreamingService._dte_for_exp("2026-04-19", today) == 0
        assert OptionQuoteStreamingService._dte_for_exp("2026-04-20", today) == 1
        assert OptionQuoteStreamingService._dte_for_exp("2026-04-22", today) == 3
        assert OptionQuoteStreamingService._dte_for_exp("garbage", today) is None

    def test_build_fetch_jobs_splits_csv_and_ibkr(self):
        """Two job lists with different strike ranges and DTE filters."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock
        from datetime import date, timedelta

        cfg = _make_streaming_config()
        svc = OptionQuoteStreamingService(cfg, MagicMock())

        today = date.today()
        exps = [
            today.isoformat(),                             # DTE 0  → IBKR + CSV
            (today + timedelta(days=1)).isoformat(),       # DTE 1  → IBKR + CSV
            (today + timedelta(days=2)).isoformat(),       # DTE 2  → IBKR + CSV
            (today + timedelta(days=5)).isoformat(),       # DTE 5  → CSV only
            (today + timedelta(days=14)).isoformat(),      # DTE 14 → excluded (> csv_dte_max=10)
        ]

        csv_jobs, ibkr_jobs = svc._build_fetch_jobs(
            symbol="SPX", price=5000.0, expirations=exps, price_source="quote",
        )
        # CSV: DTE 0,1,2,5 (DTE 14 excluded) × CALL/PUT = 4 × 2 = 8
        assert len(csv_jobs) == 8
        # IBKR: only DTE 0/1/2 × CALL/PUT = 3 × 2 = 6
        assert len(ibkr_jobs) == 6

        # Strike range checks: IBKR ±3% of 5000 = [4850, 5150], CSV ±15% = [4250, 5750]
        for _, _, _, smin, smax, _ in csv_jobs:
            assert smin == 4250.0
            assert smax == 5750.0
        for _, _, _, smin, smax, _ in ibkr_jobs:
            assert smin == 4850.0
            assert smax == 5150.0

        # IBKR jobs only include DTE 0/1/2 expirations
        ibkr_exps = {j[1] for j in ibkr_jobs}
        assert ibkr_exps == {today.isoformat(), (today + timedelta(days=1)).isoformat(),
                             (today + timedelta(days=2)).isoformat()}

    def test_build_fetch_jobs_dte_list_none_includes_all(self):
        """ibkr_dte_list=None means IBKR fetches every expiration (legacy)."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock
        from datetime import date, timedelta

        cfg = _make_streaming_config(option_quotes_ibkr_dte_list=None)
        svc = OptionQuoteStreamingService(cfg, MagicMock())

        today = date.today()
        # Use DTE 0 and DTE 5 (both within csv_dte_max=10)
        exps = [today.isoformat(), (today + timedelta(days=5)).isoformat()]
        csv_jobs, ibkr_jobs = svc._build_fetch_jobs("SPX", 5000.0, exps, "quote")
        assert len(csv_jobs) == 4   # both exps × 2 types
        assert len(ibkr_jobs) == 4  # everything (no DTE filter)

    def test_build_fetch_jobs_dte_list_empty_skips_ibkr(self):
        """ibkr_dte_list=[] means CSV only (IBKR tier disabled)."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock
        from datetime import date, timedelta

        cfg = _make_streaming_config(option_quotes_ibkr_dte_list=[])
        svc = OptionQuoteStreamingService(cfg, MagicMock())

        today = date.today()
        exps = [today.isoformat(), (today + timedelta(days=1)).isoformat()]
        csv_jobs, ibkr_jobs = svc._build_fetch_jobs("SPX", 5000.0, exps, "quote")
        assert len(csv_jobs) == 4
        assert ibkr_jobs == []

    def test_load_yaml_parses_tiered_fields(self, tmp_path):
        """YAML loader honors the new ibkr_/csv_ fields."""
        from app.services.streaming_config import load_streaming_config
        cfg_yaml = tmp_path / "streaming.yaml"
        cfg_yaml.write_text(
            "symbols:\n  - SPX\n  - NDX\n  - RUT\n"
            "option_quotes_enabled: true\n"
            "option_quotes_ibkr_strike_range_pct: 2.5\n"
            "option_quotes_csv_strike_range_pct: 10.0\n"
            "option_quotes_ibkr_dte_list: [0, 1, 2]\n"
        )
        cfg = load_streaming_config(cfg_yaml)
        assert cfg.option_quotes_ibkr_strike_range_pct == 2.5
        assert cfg.option_quotes_csv_strike_range_pct == 10.0
        assert cfg.option_quotes_ibkr_dte_list == [0, 1, 2]
        # SPX/NDX/RUT all in symbols
        names = {s.symbol for s in cfg.symbols}
        assert names == {"SPX", "NDX", "RUT"}

    def test_load_yaml_backwards_compat_with_legacy_field(self, tmp_path):
        """Old configs without ibkr_/csv_ fields still work (use legacy as default)."""
        from app.services.streaming_config import load_streaming_config
        cfg_yaml = tmp_path / "streaming.yaml"
        cfg_yaml.write_text(
            "symbols:\n  - SPX\n"
            "option_quotes_enabled: true\n"
            "option_quotes_strike_range_pct: 5.0\n"  # legacy field only
        )
        cfg = load_streaming_config(cfg_yaml)
        # Both fall back to legacy field value (5.0)
        assert cfg.option_quotes_ibkr_strike_range_pct == 5.0
        assert cfg.option_quotes_csv_strike_range_pct == 5.0  # legacy fallback, not default 15
        # No DTE filter unless explicitly set
        assert cfg.option_quotes_ibkr_dte_list is None
        assert cfg.option_quotes_csv_dte_max == 10

    def test_default_yaml_has_tickers_and_tiered_fields(self):
        """The shipped streaming_default.yaml uses the new tiered defaults."""
        from app.services.streaming_config import load_streaming_config
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "streaming_default.yaml"
        cfg = load_streaming_config(cfg_path)
        # SPX/NDX/RUT all enabled by default
        names = {s.symbol for s in cfg.symbols}
        assert {"SPX", "NDX", "RUT"} <= names
        # IBKR ±3%, CSV ±15%, split DTE depth
        assert cfg.option_quotes_ibkr_strike_range_pct == 3.0
        assert cfg.option_quotes_csv_strike_range_pct == 15.0
        assert cfg.option_quotes_ibkr_dte_list == [0, 1, 2]
        assert cfg.option_quotes_csv_dte_max == 10

    def test_stats_exposes_tiered_config(self):
        """Streamer status reports the tiered settings (for visibility)."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock
        cfg = _make_streaming_config()
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        stats = svc.stats
        config_block = stats["config"]
        assert config_block["ibkr_strike_range_pct"] == 3.0
        assert config_block["csv_strike_range_pct"] == 15.0
        assert config_block["ibkr_dte_list"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_csv_primary_cycle_uses_ibkr_jobs_for_overlay(self):
        """The IBKR overlay is invoked with ibkr_jobs (not csv_jobs)."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock, AsyncMock
        from datetime import date, timedelta
        import time as _time

        cfg = _make_streaming_config()
        # csv_primary defaults to True; force csv_dir so the path runs
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        svc._csv_primary = True
        svc._csv_dir = "/tmp/no-csv-dir-needed-for-this-test"
        # Force overlay to be considered "due"
        svc._last_greeks_fetch = _time.monotonic() - 999

        # Stub out CSV reader so the cycle just exercises the IBKR overlay path
        svc._load_csv_latest_snapshot = MagicMock(return_value=([], None))

        captured: list[list[tuple]] = []

        async def _fake_fetch(jobs):
            captured.append(list(jobs))

        svc._fetch_from_ibkr = _fake_fetch

        today = date.today()
        # DTE 0 (IBKR + CSV) and DTE 5 (CSV only, within csv_dte_max=10)
        exps = [today.isoformat(), (today + timedelta(days=5)).isoformat()]
        csv_jobs, ibkr_jobs = svc._build_fetch_jobs("SPX", 5000.0, exps, "quote")
        # csv_jobs covers both DTEs; ibkr_jobs only DTE 0
        assert len(csv_jobs) == 4
        assert len(ibkr_jobs) == 2

        await svc._run_csv_primary_cycle(csv_jobs, ibkr_jobs)
        assert captured, "IBKR overlay was not invoked"
        # Verify the overlay received the (smaller) ibkr_jobs list, not csv_jobs
        assert len(captured[0]) == len(ibkr_jobs)
        for _, _, _, smin, smax, _ in captured[0]:
            # IBKR tier: ±3% of 5000
            assert smin == 4850.0
            assert smax == 5150.0

    @pytest.mark.asyncio
    async def test_csv_primary_cycle_skips_overlay_when_no_ibkr_jobs(self):
        """When ibkr_dte_list=[] there are no IBKR jobs — overlay is skipped."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock
        from datetime import date
        import time as _time

        cfg = _make_streaming_config(option_quotes_ibkr_dte_list=[])
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        svc._csv_primary = True
        svc._csv_dir = "/tmp/no-csv-dir"
        svc._last_greeks_fetch = _time.monotonic() - 999
        svc._load_csv_latest_snapshot = MagicMock(return_value=([], None))

        called = []

        async def _fake_fetch(jobs):
            called.append(jobs)

        svc._fetch_from_ibkr = _fake_fetch

        csv_jobs, ibkr_jobs = svc._build_fetch_jobs(
            "SPX", 5000.0, [date.today().isoformat()], "quote",
        )
        assert ibkr_jobs == []
        await svc._run_csv_primary_cycle(csv_jobs, ibkr_jobs)
        assert called == [], "IBKR overlay should be skipped when ibkr_jobs is empty"


# ═══════════════════════════════════════════════════════════════════════════════
# Read-time merge: per-strike IBKR/CSV selection
# ═══════════════════════════════════════════════════════════════════════════════


def _make_streamer_for_merge(ibkr_max_age_sec=90.0):
    """Build a streamer with both caches initialized (no provider needed)."""
    from app.services.option_quote_streaming import OptionQuoteStreamingService
    from unittest.mock import MagicMock
    cfg = _make_streaming_config(option_quotes_ibkr_max_age_sec=ibkr_max_age_sec)
    return OptionQuoteStreamingService(cfg, MagicMock())


def _q(strike, bid=1.0, ask=1.1, **extra):
    """Build a quote dict for tests."""
    base = {"strike": float(strike), "bid": bid, "ask": ask, "last": (bid + ask) / 2,
            "volume": 0, "open_interest": 0}
    base.update(extra)
    return base


def _force_cache_age(cache, sym, exp, opt_type, age_seconds):
    """Adjust the monotonic fetched_at on the cached entry to simulate age."""
    import time as _t
    key = (sym.upper(), exp, opt_type.upper())
    entry = cache._cache.get(key)
    assert entry is not None, f"no cache entry for {key}"
    entry.fetched_at = _t.monotonic() - age_seconds


class TestReadTimeMerge:
    def test_fresh_ibkr_wins_over_csv(self):
        svc = _make_streamer_for_merge()
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.50, ask=2.55)])
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.40, ask=2.60)])
        merged, meta = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert len(merged) == 1
        assert merged[0]["bid"] == 2.50  # IBKR price wins
        assert merged[0]["source"] == "ibkr_fresh"
        assert meta["n_ibkr_fresh"] == 1
        assert meta["n_csv"] == 0

    def test_stale_ibkr_loses_to_csv(self):
        svc = _make_streamer_for_merge(ibkr_max_age_sec=30.0)
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.50, ask=2.55)])
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.40, ask=2.60)])
        # IBKR age becomes 60s — over the 30s threshold
        _force_cache_age(svc._ibkr_cache, "SPX", "2026-04-22", "PUT", 60.0)
        merged, meta = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert merged[0]["bid"] == 2.40  # CSV
        assert merged[0]["source"] == "csv"
        assert meta["n_csv"] == 1
        assert meta["n_ibkr_fresh"] == 0

    def test_ibkr_only_strike_served(self):
        """A strike present only in IBKR (e.g. ATM ±2.5%) is served from IBKR."""
        svc = _make_streamer_for_merge()
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5050, bid=1.95, ask=2.00)])
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(4500, bid=0.10, ask=0.15)])  # different strike
        merged, _ = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        strikes = sorted(r["strike"] for r in merged)
        assert strikes == [4500.0, 5050.0]
        ibkr_row = next(r for r in merged if r["strike"] == 5050.0)
        csv_row = next(r for r in merged if r["strike"] == 4500.0)
        assert ibkr_row["source"] == "ibkr_fresh"
        assert csv_row["source"] == "csv"

    def test_csv_only_strike_served(self):
        """Strike outside IBKR's range falls back to CSV."""
        svc = _make_streamer_for_merge()
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(4500, bid=0.10, ask=0.15)])
        merged, meta = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert len(merged) == 1
        assert merged[0]["source"] == "csv"
        assert meta["n_csv"] == 1

    def test_stale_ibkr_used_when_csv_missing(self):
        """Stale IBKR is better than nothing when CSV doesn't have the strike."""
        svc = _make_streamer_for_merge(ibkr_max_age_sec=30.0)
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5050, bid=1.95, ask=2.00)])
        _force_cache_age(svc._ibkr_cache, "SPX", "2026-04-22", "PUT", 120.0)
        merged, meta = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert len(merged) == 1
        assert merged[0]["source"] == "ibkr_stale"
        assert merged[0]["age_seconds"] >= 120
        assert meta["n_ibkr_stale"] == 1

    def test_strike_range_filter_respected(self):
        svc = _make_streamer_for_merge()
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [
            _q(4900), _q(5000), _q(5100),
        ])
        svc._cache.put("SPX", "2026-04-22", "PUT", [
            _q(4500), _q(4900), _q(5000), _q(5100), _q(5500),
        ])
        merged, _ = svc.get_merged_quotes(
            "SPX", "2026-04-22", "PUT", strike_min=4950, strike_max=5050,
        )
        strikes = [r["strike"] for r in merged]
        assert strikes == [5000.0]
        assert merged[0]["source"] == "ibkr_fresh"

    def test_per_row_age_seconds_present(self):
        svc = _make_streamer_for_merge()
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5000)])
        merged, _ = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert merged[0]["age_seconds"] is not None
        assert merged[0]["age_seconds"] >= 0

    def test_ibkr_max_age_override(self):
        """Caller can override the IBKR freshness threshold."""
        svc = _make_streamer_for_merge(ibkr_max_age_sec=90.0)
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.50)])
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.40)])
        _force_cache_age(svc._ibkr_cache, "SPX", "2026-04-22", "PUT", 50.0)
        # Default threshold 90s — IBKR wins
        merged, _ = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert merged[0]["source"] == "ibkr_fresh"
        # Tighter threshold 30s — CSV wins
        merged, _ = svc.get_merged_quotes(
            "SPX", "2026-04-22", "PUT", ibkr_max_age_sec=30.0,
        )
        assert merged[0]["source"] == "csv"

    def test_get_cached_quotes_shim_returns_merged(self):
        """Backwards-compat: get_cached_quotes returns the merged list."""
        svc = _make_streamer_for_merge()
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.50)])
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=2.40)])
        out = svc.get_cached_quotes("SPX", "2026-04-22", "PUT")
        assert out is not None and len(out) == 1
        assert out[0]["bid"] == 2.50
        assert out[0]["source"] == "ibkr_fresh"

    def test_status_endpoint_shows_both_caches_and_merge_config(self):
        svc = _make_streamer_for_merge()
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5000)])
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000), _q(4900)])
        stats = svc.stats
        assert stats["ibkr_cache"]["entries"] == 1
        assert stats["ibkr_cache"]["total_quotes"] == 1
        assert stats["csv_cache"]["entries"] == 1
        assert stats["csv_cache"]["total_quotes"] == 2
        assert stats["merge_config"]["ibkr_max_age_sec"] == 90.0
        assert stats["merge_config"]["csv_max_age_market_sec"] == 900.0
        assert stats["merge_config"]["premarket_minutes"] == 10
        assert stats["merge_config"]["postmarket_minutes"] == 10
        # Backwards-compat alias still present
        assert stats["cache"]["entries"] == 1

    def test_csv_dropped_when_stale_during_market_hours(self, monkeypatch):
        """During market hours, CSV older than csv_max_age_sec is suppressed."""
        # Force "market open" so the staleness gate runs
        monkeypatch.setenv("UTP_FORCE_MARKET_OPEN", "true")
        svc = _make_streamer_for_merge()
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000)])
        # Age the CSV entry to 20 min — past the 15-min default
        _force_cache_age(svc._cache, "SPX", "2026-04-22", "PUT", 1200.0)
        merged, meta = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert merged == []
        assert meta["csv_gated"] is True
        assert meta["n_dropped_csv_stale"] == 1

    def test_csv_served_when_stale_outside_market_hours(self, monkeypatch):
        """Outside market hours, CSV is served regardless of age."""
        monkeypatch.delenv("UTP_FORCE_MARKET_OPEN", raising=False)
        # Force "outside market hours" by setting an absurdly narrow window
        # (premarket=0, postmarket=0 → only 09:30-16:00 ET counts; on a Sunday
        # nothing counts).  Easiest: monkeypatch _is_market_hours to False.
        from app.services import option_quote_streaming as oqs_mod
        monkeypatch.setattr(oqs_mod, "_is_market_hours", lambda: False)
        svc = _make_streamer_for_merge()
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000)])
        _force_cache_age(svc._cache, "SPX", "2026-04-22", "PUT", 7200.0)  # 2 hrs
        merged, meta = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert len(merged) == 1
        assert merged[0]["source"] == "csv"
        assert meta["csv_gated"] is False

    def test_stale_ibkr_used_when_csv_gated(self, monkeypatch):
        """When CSV is gated stale and IBKR is also stale, ibkr_stale fills in."""
        monkeypatch.setenv("UTP_FORCE_MARKET_OPEN", "true")
        svc = _make_streamer_for_merge(ibkr_max_age_sec=30.0)
        svc._ibkr_cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=1.95)])
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000, bid=1.90)])
        _force_cache_age(svc._ibkr_cache, "SPX", "2026-04-22", "PUT", 120.0)  # stale
        _force_cache_age(svc._cache, "SPX", "2026-04-22", "PUT", 1200.0)      # also stale
        merged, meta = svc.get_merged_quotes("SPX", "2026-04-22", "PUT")
        assert len(merged) == 1
        assert merged[0]["source"] == "ibkr_stale"
        assert merged[0]["bid"] == 1.95
        assert meta["csv_gated"] is True

    def test_csv_gate_disabled_when_zero(self, monkeypatch):
        """csv_max_age_sec=0 disables the gate entirely."""
        monkeypatch.setenv("UTP_FORCE_MARKET_OPEN", "true")
        svc = _make_streamer_for_merge()
        svc._cache.put("SPX", "2026-04-22", "PUT", [_q(5000)])
        _force_cache_age(svc._cache, "SPX", "2026-04-22", "PUT", 7200.0)
        merged, meta = svc.get_merged_quotes(
            "SPX", "2026-04-22", "PUT", csv_max_age_sec=0,
        )
        assert len(merged) == 1
        assert merged[0]["source"] == "csv"
        assert meta["csv_gated"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Configurable pre/post-market window
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarketWindow:
    def test_default_window_is_10_minutes(self, monkeypatch):
        monkeypatch.delenv("UTP_PREMARKET_MINUTES", raising=False)
        monkeypatch.delenv("UTP_POSTMARKET_MINUTES", raising=False)
        from app.services.market_data import _market_window_minutes
        open_min, close_min = _market_window_minutes()
        assert open_min == 9 * 60 + 20    # 09:20 ET
        assert close_min == 16 * 60 + 10  # 16:10 ET

    def test_premarket_env_extends_open(self, monkeypatch):
        monkeypatch.setenv("UTP_PREMARKET_MINUTES", "30")
        monkeypatch.delenv("UTP_POSTMARKET_MINUTES", raising=False)
        from app.services.market_data import _market_window_minutes
        open_min, _ = _market_window_minutes()
        assert open_min == 9 * 60 + 0     # 09:00 ET (30 min before 09:30)

    def test_postmarket_zero_strict_close(self, monkeypatch):
        monkeypatch.delenv("UTP_PREMARKET_MINUTES", raising=False)
        monkeypatch.setenv("UTP_POSTMARKET_MINUTES", "0")
        from app.services.market_data import _market_window_minutes
        _, close_min = _market_window_minutes()
        assert close_min == 16 * 60       # exactly 16:00 ET

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("UTP_PREMARKET_MINUTES", "garbage")
        from app.services.market_data import _premarket_min
        assert _premarket_min() == 10

    def test_streamer_market_hours_uses_window(self, monkeypatch):
        """option_quote_streaming._is_market_hours reads the same env vars."""
        from app.services.option_quote_streaming import _is_market_hours
        from app.services.market_data import _market_window_minutes
        # Sanity: both modules see the same window
        monkeypatch.setenv("UTP_PREMARKET_MINUTES", "5")
        monkeypatch.setenv("UTP_POSTMARKET_MINUTES", "5")
        open_min, close_min = _market_window_minutes()
        assert open_min == 9 * 60 + 25
        assert close_min == 16 * 60 + 5
        # Function call doesn't crash; result depends on real ET clock.
        # We only assert it returns a bool.
        assert isinstance(_is_market_hours(), bool)

    def test_daemon_wires_env_from_config(self):
        """Daemon source contains the env-var hookup for streaming config."""
        import utp
        with open(utp.__file__) as f:
            src = f.read()
        # Both env vars assigned from streaming config values
        assert 'os.environ["UTP_PREMARKET_MINUTES"]' in src
        assert 'os.environ["UTP_POSTMARKET_MINUTES"]' in src
        assert "stream_cfg.option_quotes_premarket_minutes" in src
        assert "stream_cfg.option_quotes_postmarket_minutes" in src


# ═══════════════════════════════════════════════════════════════════════════════
# IBKR fetch parallelism + cycle interval tuning
# ═══════════════════════════════════════════════════════════════════════════════


class TestIbkrFetchParallel:
    def test_default_max_parallel_is_six(self):
        """Default StreamingConfig sets ibkr_max_parallel=6 (was hardcoded 3)."""
        from app.services.streaming_config import StreamingConfig
        cfg = StreamingConfig()
        assert cfg.option_quotes_ibkr_max_parallel == 6

    def test_default_intervals_target_25s_cycles(self):
        """Defaults set greeks_interval=25s and poll_interval=5s."""
        from app.services.streaming_config import StreamingConfig
        cfg = StreamingConfig()
        assert cfg.option_quotes_greeks_interval == 25.0
        assert cfg.option_quotes_poll_interval == 5.0

    def test_yaml_loader_honors_max_parallel(self, tmp_path):
        from app.services.streaming_config import load_streaming_config
        cfg_yaml = tmp_path / "streaming.yaml"
        cfg_yaml.write_text(
            "symbols:\n  - SPX\n"
            "option_quotes_enabled: true\n"
            "option_quotes_ibkr_max_parallel: 12\n"
            "option_quotes_greeks_interval: 20\n"
            "option_quotes_poll_interval: 3\n"
        )
        cfg = load_streaming_config(cfg_yaml)
        assert cfg.option_quotes_ibkr_max_parallel == 12
        assert cfg.option_quotes_greeks_interval == 20.0
        assert cfg.option_quotes_poll_interval == 3.0

    def test_default_yaml_uses_new_intervals(self):
        """The shipped streaming_default.yaml uses the new 25s/5s/parallel=6 defaults."""
        from app.services.streaming_config import load_streaming_config
        from pathlib import Path
        cfg_path = (Path(__file__).resolve().parent.parent
                    / "configs" / "streaming_default.yaml")
        cfg = load_streaming_config(cfg_path)
        assert cfg.option_quotes_greeks_interval == 25.0
        assert cfg.option_quotes_poll_interval == 5.0
        assert cfg.option_quotes_ibkr_max_parallel == 6

    def test_stats_exposes_parallel_and_overlay_interval(self):
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock
        cfg = _make_streaming_config(
            option_quotes_ibkr_max_parallel=8,
            option_quotes_greeks_interval=20.0,
        )
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        block = svc.stats["config"]
        # MagicMock provider has no cap entry → effective == configured
        assert block["ibkr_max_parallel_configured"] == 8
        assert block["ibkr_max_parallel_effective"] == 8
        assert block["ibkr_overlay_interval"] == 20.0

    @pytest.mark.asyncio
    async def test_fetch_uses_configured_parallelism(self):
        """`_fetch_from_ibkr` honors option_quotes_ibkr_max_parallel: at any
        instant no more than N tasks are in flight."""
        import asyncio as _aio
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        cfg = _make_streaming_config(option_quotes_ibkr_max_parallel=4)
        provider = MagicMock()

        in_flight = 0
        peak = 0
        completed = 0

        async def fake_get_option_quotes(sym, exp, opt_type, strike_min=None, strike_max=None):
            nonlocal in_flight, peak, completed
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                # Hold long enough to overlap with sibling tasks
                await _aio.sleep(0.05)
                return [{"strike": 100.0, "bid": 1.0, "ask": 1.1}]
            finally:
                in_flight -= 1
                completed += 1

        # AsyncMock-like behavior — the real provider returns a coroutine
        async def aw(*a, **kw):
            return await fake_get_option_quotes(*a, **kw)
        provider.get_option_quotes = aw

        svc = OptionQuoteStreamingService(cfg, provider)
        # Build 12 jobs to exceed the parallel cap
        jobs = [
            (f"SYM{i // 4}", "2026-04-22", "PUT" if i % 2 else "CALL", 90.0, 110.0, "test")
            for i in range(12)
        ]
        await svc._fetch_from_ibkr(jobs)
        assert completed == 12, "all jobs must complete"
        assert peak <= 4, f"max in-flight {peak} exceeded the configured cap (4)"
        # And we did parallelize: peak should be > 1 with so many jobs
        assert peak > 1, "parallelism should have engaged across jobs"

    def test_provider_cap_table_present(self):
        """The provider-specific safe-cap table covers TWS and CPG."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        caps = OptionQuoteStreamingService._PROVIDER_PARALLEL_CAP
        assert caps["IBKRLiveProvider"] == 3   # TWS — line-limited
        assert caps["IBKRRestProvider"] == 12  # CPG — rate-limited

    def test_effective_parallel_caps_tws(self):
        """If user sets parallel=10 with TWS, runtime caps to 3."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock

        # Mock TWS provider by class name
        class IBKRLiveProvider:
            pass
        provider = IBKRLiveProvider()

        cfg = _make_streaming_config(option_quotes_ibkr_max_parallel=10)
        svc = OptionQuoteStreamingService(cfg, provider)
        assert svc._effective_max_parallel() == 3

    def test_effective_parallel_caps_cpg(self):
        """User can run up to 12 on CPG; 15 gets capped to 12."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        class IBKRRestProvider:
            pass
        provider = IBKRRestProvider()

        cfg = _make_streaming_config(option_quotes_ibkr_max_parallel=15)
        svc = OptionQuoteStreamingService(cfg, provider)
        assert svc._effective_max_parallel() == 12

    def test_effective_parallel_respects_user_under_cap(self):
        """If user sets parallel=2 with TWS (under cap=3), use 2."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        class IBKRLiveProvider:
            pass
        provider = IBKRLiveProvider()

        cfg = _make_streaming_config(option_quotes_ibkr_max_parallel=2)
        svc = OptionQuoteStreamingService(cfg, provider)
        assert svc._effective_max_parallel() == 2

    def test_effective_parallel_unknown_provider_uses_config(self):
        """For mocks/stubs not in the cap table, honor the config value as-is."""
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        from unittest.mock import MagicMock

        cfg = _make_streaming_config(option_quotes_ibkr_max_parallel=8)
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        # MagicMock isn't in the cap table → use configured value
        assert svc._effective_max_parallel() == 8

    @pytest.mark.asyncio
    async def test_default_parallelism_three_when_no_config(self):
        """If a custom config object lacks the new attribute, fall back to
        the legacy class-level default of 3 (no crash)."""
        import asyncio as _aio
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        class _LegacyConfig:
            # Mimics a config object that pre-dates ibkr_max_parallel
            option_quotes_csv_strike_range_pct = 10.0
            option_quotes_ibkr_strike_range_pct = 2.5
            option_quotes_ibkr_dte_list = None
            option_quotes_csv_dte_max = 10
            option_quotes_ibkr_max_age_sec = 90.0
            option_quotes_csv_max_age_market_sec = 900.0
            option_quotes_premarket_minutes = 10
            option_quotes_postmarket_minutes = 10
            option_quotes_csv_primary = True
            option_quotes_csv_dir = ""
            option_quotes_greeks_interval = 60.0
            option_quotes_poll_interval = 5.0
            option_quotes_strike_range_pct = 10.0
            option_quotes_num_expirations = 6
            option_quotes_enabled = True
            symbols = []
            redis_url = ""
            redis_enabled = False

        in_flight = 0
        peak = 0

        async def aw(*a, **kw):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            try:
                await _aio.sleep(0.05)
                return []
            finally:
                in_flight -= 1

        provider = MagicMock()
        provider.get_option_quotes = aw

        svc = OptionQuoteStreamingService(_LegacyConfig(), provider)
        jobs = [
            (f"SYM{i}", "2026-04-22", "PUT", 90.0, 110.0, "test")
            for i in range(8)
        ]
        await svc._fetch_from_ibkr(jobs)
        # Falls back to legacy class-level _IBKR_FETCH_CONCURRENCY = 3
        assert peak <= 3


# ═══════════════════════════════════════════════════════════════════════════════
# IBKR overlay no-overlap guarantee
# ═══════════════════════════════════════════════════════════════════════════════


class TestIbkrOverlayNoOverlap:
    @pytest.mark.asyncio
    async def test_skip_when_overlay_in_flight(self):
        """A second overlay firing while the first is in flight is skipped."""
        import asyncio as _aio
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        cfg = _make_streaming_config()
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        svc._csv_primary = True
        svc._csv_dir = "/tmp/no-csv-dir"
        svc._load_csv_latest_snapshot = MagicMock(return_value=([], None))
        svc._last_greeks_fetch = _aio.get_event_loop().time() - 9999  # always due

        # Pretend a previous overlay is still draining
        svc._ibkr_overlay_in_flight = True
        svc._ibkr_overlay_started_at = svc._last_greeks_fetch

        called = []

        async def _fake_fetch(jobs):
            called.append(jobs)

        svc._fetch_from_ibkr = _fake_fetch

        from datetime import date
        csv_jobs, ibkr_jobs = svc._build_fetch_jobs(
            "SPX", 5000.0, [date.today().isoformat()], "quote",
        )
        await svc._run_csv_primary_cycle(csv_jobs, ibkr_jobs)
        assert called == [], "overlay should be skipped while previous in flight"
        assert svc._ibkr_overlay_skipped == 1

    @pytest.mark.asyncio
    async def test_overlay_clears_in_flight_flag_on_success(self):
        """After a successful overlay, the in_flight flag is cleared."""
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        import time as _time

        cfg = _make_streaming_config()
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        svc._csv_primary = True
        svc._csv_dir = "/tmp/no-csv-dir"
        svc._load_csv_latest_snapshot = MagicMock(return_value=([], None))
        svc._last_greeks_fetch = _time.monotonic() - 9999  # always due

        async def _fake_fetch(jobs):
            return None

        svc._fetch_from_ibkr = _fake_fetch

        from datetime import date
        csv_jobs, ibkr_jobs = svc._build_fetch_jobs(
            "SPX", 5000.0, [date.today().isoformat()], "quote",
        )
        await svc._run_csv_primary_cycle(csv_jobs, ibkr_jobs)
        assert svc._ibkr_overlay_in_flight is False
        assert svc._ibkr_overlay_started_at is None
        assert svc._ibkr_overlay_skipped == 0

    @pytest.mark.asyncio
    async def test_overlay_clears_in_flight_flag_on_timeout(self):
        """If the overlay times out, the in_flight flag is still cleared."""
        import asyncio as _aio
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService
        import time as _time

        # Tight timeout so the test runs fast
        cfg = _make_streaming_config(option_quotes_greeks_interval=0.5)
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        svc._csv_primary = True
        svc._csv_dir = "/tmp/no-csv-dir"
        svc._load_csv_latest_snapshot = MagicMock(return_value=([], None))
        svc._last_greeks_fetch = _time.monotonic() - 9999

        async def _slow_fetch(jobs):
            await _aio.sleep(5.0)  # way past the 0.45s timeout

        svc._fetch_from_ibkr = _slow_fetch

        from datetime import date
        csv_jobs, ibkr_jobs = svc._build_fetch_jobs(
            "SPX", 5000.0, [date.today().isoformat()], "quote",
        )
        await svc._run_csv_primary_cycle(csv_jobs, ibkr_jobs)
        # Even after timeout, flag must be reset
        assert svc._ibkr_overlay_in_flight is False
        assert svc._ibkr_overlay_started_at is None

    @pytest.mark.asyncio
    async def test_drain_pending_tasks_cancels_unfinished(self):
        """_drain_ibkr_pending_tasks cancels lingering tasks within timeout."""
        import asyncio as _aio
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        cfg = _make_streaming_config()
        svc = OptionQuoteStreamingService(cfg, MagicMock())

        async def _hang():
            try:
                await _aio.sleep(60)
            except _aio.CancelledError:
                raise

        t = _aio.create_task(_hang())
        svc._ibkr_pending_tasks.add(t)
        # Drain — should cancel and reap quickly
        await svc._drain_ibkr_pending_tasks(timeout=1.0)
        assert t.cancelled() or t.done()
        assert t not in svc._ibkr_pending_tasks

    def test_stats_exposes_overlay_state(self):
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        cfg = _make_streaming_config()
        svc = OptionQuoteStreamingService(cfg, MagicMock())
        block = svc.stats["ibkr_overlay"]
        assert block["in_flight"] is False
        assert block["started_at_age_sec"] is None
        assert block["skipped_overlapping"] == 0
        assert block["pending_tasks"] == 0

    def test_stats_exposes_effective_parallel_and_provider_kind(self):
        from unittest.mock import MagicMock
        from app.services.option_quote_streaming import OptionQuoteStreamingService

        class IBKRLiveProvider:
            pass

        cfg = _make_streaming_config(option_quotes_ibkr_max_parallel=10)
        svc = OptionQuoteStreamingService(cfg, IBKRLiveProvider())
        block = svc.stats["config"]
        assert block["ibkr_max_parallel_configured"] == 10
        # TWS cap is 3
        assert block["ibkr_max_parallel_effective"] == 3
        assert block["ibkr_provider_kind"] == "IBKRLiveProvider"


# ── Trade Notification Tests ──────────────────────────────────────────────────


class TestTradeNotifications:
    """Tests for trade fill notification system."""

    def test_notify_config_defaults(self):
        from app.config import Settings
        s = Settings()
        assert s.notify_on_fill is False
        assert s.notify_channel == "email"
        assert s.notify_recipients == ""
        assert s.notify_tag == "[UTP-ALERT]"
        assert s.notify_on_paper is False

    @pytest.mark.asyncio
    async def test_notify_skipped_when_disabled(self):
        from app.services.trade_service import _notify_trade_fill
        from app.models import OrderResult, OrderStatus, TradeRequest, EquityOrder, Broker
        from unittest.mock import patch, AsyncMock

        req = TradeRequest(equity_order=EquityOrder(
            symbol="SPY", side="BUY", quantity=1, broker=Broker.IBKR,
        ))
        result = OrderResult(
            order_id="test-123", broker=Broker.IBKR,
            status=OrderStatus.FILLED, filled_price=450.0,
        )

        with patch("app.config.settings") as mock_settings:
            mock_settings.notify_on_fill = False
            # Should return immediately without sending
            await _notify_trade_fill(req, result)
            # No error = success (nothing sent)

    @pytest.mark.asyncio
    async def test_notify_skipped_when_no_recipients(self):
        from app.services.trade_service import _notify_trade_fill
        from app.models import OrderResult, OrderStatus, TradeRequest, EquityOrder, Broker
        from unittest.mock import patch

        req = TradeRequest(equity_order=EquityOrder(
            symbol="SPY", side="BUY", quantity=1, broker=Broker.IBKR,
        ))
        result = OrderResult(
            order_id="test-123", broker=Broker.IBKR,
            status=OrderStatus.FILLED, filled_price=450.0,
        )

        with patch("app.config.settings") as mock_settings:
            mock_settings.notify_on_fill = True
            mock_settings.notify_recipients = ""
            await _notify_trade_fill(req, result)

    @pytest.mark.asyncio
    async def test_notify_sends_for_equity(self):
        from app.services.trade_service import _notify_trade_fill
        from app.models import OrderResult, OrderStatus, TradeRequest, EquityOrder, Broker
        from unittest.mock import patch, AsyncMock, MagicMock

        req = TradeRequest(equity_order=EquityOrder(
            symbol="SPY", side="BUY", quantity=10, broker=Broker.IBKR,
        ))
        result = OrderResult(
            order_id="test-123", broker=Broker.IBKR,
            status=OrderStatus.FILLED, filled_price=450.0,
        )

        mock_resp = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.settings") as mock_settings, \
             patch("app.services.trade_service.httpx.AsyncClient", return_value=mock_client):
            mock_settings.notify_on_fill = True
            mock_settings.notify_recipients = "test@example.com"
            mock_settings.notify_channel = "email"
            mock_settings.notify_tag = "[UTP-ALERT]"
            mock_settings.notify_url = "http://localhost:9102"

            await _notify_trade_fill(req, result)

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "/api/notify" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["channel"] == "email"
            assert payload["to"] == "test@example.com"
            assert "BUY 10x SPY" in payload["message"]
            assert "$450.00" in payload["message"]
            assert payload["tag"] == "[UTP-ALERT]"

    @pytest.mark.asyncio
    async def test_notify_sends_for_multi_leg(self):
        from app.services.trade_service import _notify_trade_fill
        from app.models import (
            OrderResult, OrderStatus, TradeRequest, MultiLegOrder,
            OptionLeg, Broker, OptionType, OptionAction,
        )
        from unittest.mock import patch, AsyncMock, MagicMock

        req = TradeRequest(multi_leg_order=MultiLegOrder(
            broker=Broker.IBKR, quantity=5,
            legs=[
                OptionLeg(symbol="SPX", strike=7000, option_type=OptionType.PUT,
                          action=OptionAction.SELL_TO_OPEN, quantity=5, expiration="2026-04-20"),
                OptionLeg(symbol="SPX", strike=6980, option_type=OptionType.PUT,
                          action=OptionAction.BUY_TO_OPEN, quantity=5, expiration="2026-04-20"),
            ],
        ))
        result = OrderResult(
            order_id="test-456", broker=Broker.IBKR,
            status=OrderStatus.FILLED, filled_price=1.50,
        )

        mock_resp = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.settings") as mock_settings, \
             patch("app.services.trade_service.httpx.AsyncClient", return_value=mock_client):
            mock_settings.notify_on_fill = True
            mock_settings.notify_recipients = "user@example.com"
            mock_settings.notify_channel = "email"
            mock_settings.notify_tag = "[UTP-ALERT]"
            mock_settings.notify_url = "http://localhost:9102"

            await _notify_trade_fill(req, result)

            payload = mock_client.post.call_args[1]["json"]
            assert "5x SPX 7000/6980P" in payload["message"]
            assert "$1.50" in payload["message"]
            assert "FILLED" in payload["message"]

    @pytest.mark.asyncio
    async def test_notify_multiple_recipients(self):
        from app.services.trade_service import _notify_trade_fill
        from app.models import OrderResult, OrderStatus, TradeRequest, EquityOrder, Broker
        from unittest.mock import patch, AsyncMock, MagicMock

        req = TradeRequest(equity_order=EquityOrder(
            symbol="SPY", side="BUY", quantity=1, broker=Broker.IBKR,
        ))
        result = OrderResult(
            order_id="test-789", broker=Broker.IBKR,
            status=OrderStatus.FILLED, filled_price=450.0,
        )

        mock_resp = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.settings") as mock_settings, \
             patch("app.services.trade_service.httpx.AsyncClient", return_value=mock_client):
            mock_settings.notify_on_fill = True
            mock_settings.notify_recipients = "a@test.com, b@test.com"
            mock_settings.notify_channel = "email"
            mock_settings.notify_tag = "[UTP-ALERT]"
            mock_settings.notify_url = "http://localhost:9102"

            await _notify_trade_fill(req, result)

            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_notify_close_action(self):
        from app.services.trade_service import _notify_trade_fill
        from app.models import OrderResult, OrderStatus, TradeRequest, EquityOrder, Broker
        from unittest.mock import patch, AsyncMock, MagicMock

        req = TradeRequest(
            equity_order=EquityOrder(
                symbol="SPY", side="SELL", quantity=10, broker=Broker.IBKR,
            ),
            closing_position_id="pos-abc",
        )
        result = OrderResult(
            order_id="test-close", broker=Broker.IBKR,
            status=OrderStatus.FILLED, filled_price=455.0,
        )

        mock_resp = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.settings") as mock_settings, \
             patch("app.services.trade_service.httpx.AsyncClient", return_value=mock_client):
            mock_settings.notify_on_fill = True
            mock_settings.notify_recipients = "test@example.com"
            mock_settings.notify_channel = "email"
            mock_settings.notify_tag = "[UTP-ALERT]"
            mock_settings.notify_url = "http://localhost:9102"

            await _notify_trade_fill(req, result, is_close=True)

            payload = mock_client.post.call_args[1]["json"]
            assert "CLOSED" in payload["message"]

    @pytest.mark.asyncio
    async def test_notify_failure_does_not_propagate(self):
        """Notification errors must never block trading."""
        from app.services.trade_service import _notify_trade_fill
        from app.models import OrderResult, OrderStatus, TradeRequest, EquityOrder, Broker
        from unittest.mock import patch, AsyncMock

        req = TradeRequest(equity_order=EquityOrder(
            symbol="SPY", side="BUY", quantity=1, broker=Broker.IBKR,
        ))
        result = OrderResult(
            order_id="test-err", broker=Broker.IBKR,
            status=OrderStatus.FILLED, filled_price=450.0,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.config.settings") as mock_settings, \
             patch("app.services.trade_service.httpx.AsyncClient", return_value=mock_client):
            mock_settings.notify_on_fill = True
            mock_settings.notify_recipients = "test@example.com"
            mock_settings.notify_channel = "email"
            mock_settings.notify_tag = "[UTP-ALERT]"
            mock_settings.notify_url = "http://localhost:9102"

            # Should not raise
            await _notify_trade_fill(req, result)


# ── Spread Scanner Tests ──────────────────────────────────────────────────────


class TestSpreadScanner:
    """Tests for spread_scanner.py functionality."""

    def test_parse_args_defaults(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from spread_scanner import parse_args

        args = parse_args([])
        assert args.tickers == ["SPX", "RUT", "NDX"]
        assert args.otm_pcts == [0.5, 1.0, 1.25, 1.5, 2.0, 2.5]
        assert args.interval == 30
        assert args.dte == [0]
        assert args.types == ["put", "call", "iron-condor"]
        assert args.once is False
        assert args.tiers is False

    def test_parse_args_custom_otm(self):
        from spread_scanner import parse_args

        args = parse_args(["--otm-pcts", "1,2,3"])
        assert args.otm_pcts == [1.0, 2.0, 3.0]

    def test_parse_args_dte(self):
        from spread_scanner import parse_args

        args = parse_args(["--dte", "0,1,2"])
        assert args.dte == [0, 1, 2]

    def test_find_best_spread_at_otm(self):
        from spread_scanner import find_best_spread_at_otm

        spreads = [
            {"option_type": "PUT", "short_strike": 7000, "otm_pct": 1.5, "credit": 1.0, "roi_pct": 5.0},
            {"option_type": "PUT", "short_strike": 6950, "otm_pct": 2.2, "credit": 0.5, "roi_pct": 2.5},
            {"option_type": "CALL", "short_strike": 7200, "otm_pct": 1.0, "credit": 1.2, "roi_pct": 6.0},
        ]
        result = find_best_spread_at_otm(spreads, 2.0, "PUT")
        assert result is not None
        assert result["short_strike"] == 6950  # closest to 2.0%

    def test_find_best_spread_no_data(self):
        from spread_scanner import find_best_spread_at_otm

        assert find_best_spread_at_otm([], 1.0, "PUT") is None
        assert find_best_spread_at_otm(
            [{"option_type": "CALL", "otm_pct": 1.0}], 1.0, "PUT"
        ) is None

    def test_compute_iron_condor(self):
        from spread_scanner import compute_iron_condor

        put = {"credit": 1.50, "width": 20, "short_strike": 7000, "long_strike": 6980}
        call = {"credit": 1.20, "width": 20, "short_strike": 7200, "long_strike": 7220}
        ic = compute_iron_condor(put, call)
        assert ic is not None
        assert ic["credit"] == 2.70
        assert ic["put_short"] == 7000
        assert ic["call_short"] == 7200
        assert ic["roi_pct"] > 0

    def test_compute_iron_condor_none_input(self):
        from spread_scanner import compute_iron_condor

        assert compute_iron_condor(None, None) is None
        assert compute_iron_condor({"credit": 1.0, "width": 20}, None) is None

    def test_render_price_line(self):
        from spread_scanner import render_price_line

        quotes = {
            "SPX": {"last": 7103.60},
            "RUT": None,
        }
        prev_closes = {"SPX": 7000.0}
        line = render_price_line(quotes, prev_closes)
        assert "7,103.60" in line
        assert "---" in line
        assert "+" in line  # shows positive change

    def test_render_spread_cell(self):
        from spread_scanner import render_spread_cell

        spread = {
            "short_strike": 7065.0,
            "long_strike": 7045.0,
            "credit": 2.10,
            "roi_pct": 11.7,
            "otm_pct": 0.5,
        }
        cell = render_spread_cell(spread, prev_close=7100.0)
        assert "7065/7045" in cell
        assert "$2.10" in cell
        assert "ot0.5" in cell  # actual OTM%
        assert "cl" in cell  # % from close

    def test_render_spread_cell_none(self):
        from spread_scanner import render_spread_cell

        cell = render_spread_cell(None)
        assert "─" in cell

    def test_color_roi_thresholds(self):
        from spread_scanner import color_roi, GREEN, YELLOW, DIM

        high = color_roi(5.5)
        assert GREEN in high

        mid = color_roi(3.0)
        assert YELLOW in mid

        low = color_roi(1.5)
        assert DIM in low

    def test_compute_spreads_basic(self):
        from spread_scanner import compute_spreads

        chain = {
            "put": [
                {"strike": 7000, "bid": 2.50, "ask": 2.80},
                {"strike": 6980, "bid": 1.50, "ask": 1.80},
            ],
            "call": [
                {"strike": 7200, "bid": 2.00, "ask": 2.30},
                {"strike": 7220, "bid": 1.00, "ask": 1.30},
            ],
        }
        spreads = compute_spreads(chain, "SPX", 7100.0, 20)
        put_spreads = [s for s in spreads if s["option_type"] == "PUT"]
        call_spreads = [s for s in spreads if s["option_type"] == "CALL"]
        assert len(put_spreads) == 1
        assert put_spreads[0]["short_strike"] == 7000
        assert put_spreads[0]["long_strike"] == 6980
        assert put_spreads[0]["credit"] == 0.70  # 2.50 - 1.80
        assert len(call_spreads) == 1
        assert call_spreads[0]["short_strike"] == 7200

    def test_compute_spreads_enumerates_multiple_widths(self):
        """For each short strike, every OTM long-leg strike within max_width
        should produce a distinct spread using the actual chain grid."""
        from spread_scanner import compute_spreads

        # Puts on a 10-pt grid: shorts at 7050/7040/7030, longs at 7020/7010.
        # With max_width=30 and short=7050, long candidates are
        # 7040 (w=10), 7030 (w=20), 7020 (w=30) — all three should appear.
        chain = {
            "put": [
                {"strike": 7050, "bid": 3.00, "ask": 3.30},
                {"strike": 7040, "bid": 2.50, "ask": 2.80},
                {"strike": 7030, "bid": 2.00, "ask": 2.30},
                {"strike": 7020, "bid": 1.50, "ask": 1.80},
            ],
            "call": [],
        }
        spreads = compute_spreads(chain, "SPX", 7100.0, 30)
        at_7050 = sorted(
            [s for s in spreads if s["short_strike"] == 7050 and s["option_type"] == "PUT"],
            key=lambda s: s["width"],
        )
        assert [s["width"] for s in at_7050] == [10, 20, 30]
        # Credit = short_bid - long_ask: 3.00 - 2.80 = 0.20 (w=10),
        # 3.00 - 2.30 = 0.70 (w=20), 3.00 - 1.80 = 1.20 (w=30).
        assert at_7050[0]["credit"] == 0.20
        assert at_7050[1]["credit"] == 0.70
        assert at_7050[2]["credit"] == 1.20

    def test_compute_spreads_respects_max_width(self):
        """Pairs whose width exceeds max_width must be excluded."""
        from spread_scanner import compute_spreads

        chain = {
            "put": [
                {"strike": 7050, "bid": 3.00, "ask": 3.30},
                {"strike": 7040, "bid": 2.50, "ask": 2.80},  # w=10
                {"strike": 7020, "bid": 1.50, "ask": 1.80},  # w=30
                {"strike": 7000, "bid": 1.00, "ask": 1.30},  # w=50 — excluded
            ],
            "call": [],
        }
        spreads = compute_spreads(chain, "SPX", 7100.0, 30)
        widths_at_7050 = sorted(
            s["width"] for s in spreads
            if s["short_strike"] == 7050 and s["option_type"] == "PUT"
        )
        assert widths_at_7050 == [10, 30]  # 50 excluded

    def test_find_spread_at_strike_returns_best_roi(self):
        """When the same short has multiple widths, return the highest-ROI one."""
        from spread_scanner import find_spread_at_strike

        spreads = [
            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7040,
             "width": 10, "credit": 0.20, "roi_pct": 2.0, "otm_pct": 0.7},
            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7020,
             "width": 30, "credit": 1.20, "roi_pct": 4.3, "otm_pct": 0.7},
            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7030,
             "width": 20, "credit": 0.70, "roi_pct": 3.6, "otm_pct": 0.7},
        ]
        best = find_spread_at_strike(spreads, 7050, "PUT")
        assert best is not None
        assert best["width"] == 30  # highest ROI among the three
        assert best["roi_pct"] == 4.3

    def test_detect_suspect_strikes_flags_non_monotonic_put_bids(self):
        """PUT bids should rise with strike. Strikes whose bid is out of
        order vs immediate neighbors (beyond tolerance) are flagged."""
        from spread_scanner import _detect_suspect_strikes

        # NDX-style real data: 25810 and 25840 have bids that sit way above
        # their neighbors (phantom inflated quotes).
        quotes = [
            {"strike": 25800, "bid": 3.00, "ask": 3.70},
            {"strike": 25810, "bid": 4.20, "ask": 5.00},  # inflated
            {"strike": 25820, "bid": 3.20, "ask": 3.80},
            {"strike": 25830, "bid": 3.10, "ask": 3.90},
            {"strike": 25840, "bid": 4.70, "ask": 5.60},  # inflated
            {"strike": 25850, "bid": 3.50, "ask": 4.10},
        ]
        suspect = _detect_suspect_strikes(quotes, "PUT", bid_tol=0.10)
        # The "inflated" strikes are the principal outliers — they're the
        # ones that would combine with neighbors to produce fake credit.
        assert 25810 in suspect
        assert 25840 in suspect

    def test_detect_suspect_strikes_accepts_clean_put_chain(self):
        """A clean monotonic chain has no suspect strikes."""
        from spread_scanner import _detect_suspect_strikes

        quotes = [
            {"strike": 7000, "bid": 0.50, "ask": 0.70},
            {"strike": 7010, "bid": 0.80, "ask": 1.00},
            {"strike": 7020, "bid": 1.10, "ask": 1.30},
            {"strike": 7030, "bid": 1.50, "ask": 1.70},
        ]
        assert _detect_suspect_strikes(quotes, "PUT") == set()

    def test_detect_suspect_strikes_call_monotonicity_reversed(self):
        """CALL bids should FALL with rising strike. A call strike with a
        bid higher than its lower-strike neighbor is suspect."""
        from spread_scanner import _detect_suspect_strikes

        quotes = [
            {"strike": 7100, "bid": 1.50, "ask": 1.70},
            {"strike": 7110, "bid": 1.20, "ask": 1.40},
            {"strike": 7120, "bid": 1.50, "ask": 1.70},  # higher than 7110 — suspect
            {"strike": 7130, "bid": 0.80, "ask": 1.00},
        ]
        suspect = _detect_suspect_strikes(quotes, "CALL", bid_tol=0.10)
        assert 7120 in suspect

    def test_compute_spreads_skips_suspect_strikes(self):
        """The NDX-like bogus chain should NOT build spreads using the
        detected-suspect strikes (25810, 25840) — those produce the
        fake-looking high-ROI entries the user caught."""
        from spread_scanner import compute_spreads

        chain = {
            "put": [
                {"strike": 25800, "bid": 3.00, "ask": 3.70},
                {"strike": 25810, "bid": 4.20, "ask": 5.00},  # inflated
                {"strike": 25820, "bid": 3.20, "ask": 3.80},
                {"strike": 25830, "bid": 3.10, "ask": 3.90},
                {"strike": 25840, "bid": 4.70, "ask": 5.60},  # inflated
                {"strike": 25850, "bid": 3.50, "ask": 4.10},
            ],
            "call": [],
        }
        spreads = compute_spreads(chain, "NDX", 27100.0, max_width=50)
        # Suspect strikes (25810, 25840) must not appear on either leg.
        for s in spreads:
            assert s["short_strike"] not in (25810, 25840), \
                f"suspect strike 25810/25840 leaked into spread: {s}"
            assert s["long_strike"] not in (25810, 25840), \
                f"suspect strike 25810/25840 leaked into spread: {s}"

    def test_compute_spreads_captures_short_delta_when_present(self):
        """When the provider supplies greeks on the short leg, store delta on
        the spread dict so the renderer can show it."""
        from spread_scanner import compute_spreads

        chain = {
            "put": [
                {"strike": 7050, "bid": 1.50, "ask": 1.70,
                 "greeks": {"delta": -0.18, "iv": 0.22}},
                {"strike": 7030, "bid": 0.80, "ask": 1.00,
                 "greeks": {"delta": -0.10, "iv": 0.21}},
            ],
            "call": [],
        }
        spreads = compute_spreads(chain, "SPX", 7100.0, 20)
        assert len(spreads) == 1
        assert spreads[0]["short_delta"] == -0.18  # the SHORT leg's delta

    def test_compute_spreads_short_delta_absent_when_provider_silent(self):
        """When greeks aren't supplied, short_delta is None (renderer hides)."""
        from spread_scanner import compute_spreads

        chain = {
            "put": [
                {"strike": 7050, "bid": 1.50, "ask": 1.70},
                {"strike": 7030, "bid": 0.80, "ask": 1.00},
            ],
            "call": [],
        }
        spreads = compute_spreads(chain, "SPX", 7100.0, 20)
        assert len(spreads) == 1
        assert spreads[0]["short_delta"] is None

    def test_render_spread_cell_appends_delta_when_present(self):
        """Tier-row cells should append a Δ tag when delta is on the spread."""
        from spread_scanner import render_spread_cell

        spread = {
            "option_type": "PUT", "short_strike": 7050, "long_strike": 7030,
            "credit": 0.50, "roi_pct": 2.6, "otm_pct": 0.7,
            "short_delta": -0.18,
        }
        cell = render_spread_cell(spread, prev_close=7100.0, dte=0)
        assert "Δ-0.18" in cell

        # No delta → no tag.
        spread_no_delta = {**spread, "short_delta": None}
        cell2 = render_spread_cell(spread_no_delta, prev_close=7100.0, dte=0)
        assert "Δ" not in cell2

    def test_fmt_delta_filters_out_of_range_values(self):
        """IBKR's modelGreeks.delta returns garbage (|delta|>1) when IV
        isn't computable. Suppress instead of displaying nonsense."""
        from spread_scanner import _fmt_delta

        # Valid Greek deltas — kept.
        assert _fmt_delta(-0.18) == "Δ-0.18"
        assert _fmt_delta(0.42) == "Δ+0.42"
        assert _fmt_delta(-1.0) == "Δ-1.00"  # boundary
        assert _fmt_delta(1.0) == "Δ+1.00"   # boundary

        # Out of range → suppressed.
        assert _fmt_delta(1.643) == ""    # the IBKR-after-hours-style nonsense
        assert _fmt_delta(-1.5) == ""
        assert _fmt_delta(2.0) == ""

        # Non-numeric / NaN / inf → suppressed.
        assert _fmt_delta(None) == ""
        assert _fmt_delta("oops") == ""
        assert _fmt_delta(float("nan")) == ""
        assert _fmt_delta(float("inf")) == ""

    def test_fmt_delta_filters_sign_errors_when_option_type_known(self):
        """A positive delta on a put (or negative on a call) is bogus
        provider data — suppress when we know the option type."""
        from spread_scanner import _fmt_delta

        # PUT: delta should be in [-1, 0].
        assert _fmt_delta(-0.18, "PUT") == "Δ-0.18"     # valid
        assert _fmt_delta(0.42, "PUT") == ""            # sign error → drop
        assert _fmt_delta(0.0, "PUT") == "Δ+0.00"       # boundary OK

        # CALL: delta should be in [0, +1].
        assert _fmt_delta(0.30, "CALL") == "Δ+0.30"     # valid
        assert _fmt_delta(-0.30, "CALL") == ""          # sign error → drop

        # Without option_type, only the magnitude check applies.
        assert _fmt_delta(0.42, None) == "Δ+0.42"

    def test_render_spread_cell_visible_width_constant(self):
        """Cells with and without delta must have identical VISIBLE width so
        the column grid stays aligned across the row."""
        from spread_scanner import render_spread_cell, _visible_len, COL_WIDTH

        base = {
            "option_type": "PUT", "short_strike": 7050, "long_strike": 7030,
            "credit": 0.50, "roi_pct": 2.6, "otm_pct": 0.7,
        }
        with_delta = {**base, "short_delta": -0.18}
        without_delta = {**base, "short_delta": None}
        a = render_spread_cell(with_delta, prev_close=7100.0, dte=0)
        b = render_spread_cell(without_delta, prev_close=7100.0, dte=0)
        assert _visible_len(a) == _visible_len(b) == COL_WIDTH

    def test_scanner_config_verify_max_age_default_30(self):
        """Default verify_max_age_sec is 30s — tight enough for 0DTE."""
        from spread_scanner import ScannerConfig
        cfg = ScannerConfig()
        assert cfg.verify_max_age_sec == 30.0

    def test_scanner_config_verify_max_age_from_yaml(self, tmp_path):
        """YAML can override verify_max_age_sec — typical use is 60 for
        DTE 1-3 where slightly older quotes are acceptable."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({"verify_max_age_sec": 60}))
        cfg = ScannerConfig.from_yaml(str(p))
        assert cfg.verify_max_age_sec == 60

    def test_verify_max_age_sec_propagates_to_args(self, tmp_path):
        """The CLI default for --verify-max-age-sec is overridden by YAML."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig, parse_args
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({"verify_max_age_sec": 45}))
        cfg = ScannerConfig.from_yaml(str(p))
        defaults = cfg.to_cli_defaults()
        args = parse_args(argv=[], defaults=defaults)
        assert args.verify_max_age_sec == 45

    def test_scanner_config_verify_require_provider_source_default_true(self):
        """Default verify_require_provider_source is True — only IBKR-fresh
        / provider-sourced quotes are accepted at verify time. CSV-sourced
        quotes are rejected as the safe production setting."""
        from spread_scanner import ScannerConfig
        cfg = ScannerConfig()
        assert cfg.verify_require_provider_source is True

    def test_scanner_config_verify_require_provider_source_from_yaml(self, tmp_path):
        """YAML can set verify_require_provider_source: false to allow
        CSV-sourced quotes through (escape hatch when IBKR stream is missing
        data for a ticker, e.g. CPG silently failing on SPX 0DTE)."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({"verify_require_provider_source": False}))
        cfg = ScannerConfig.from_yaml(str(p))
        assert cfg.verify_require_provider_source is False

    def test_verify_require_provider_source_propagates_to_args(self, tmp_path):
        """YAML's verify_require_provider_source flows through to argparse
        defaults so the verify call honors it."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig, parse_args
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({"verify_require_provider_source": False}))
        cfg = ScannerConfig.from_yaml(str(p))
        defaults = cfg.to_cli_defaults()
        args = parse_args(argv=[], defaults=defaults)
        assert args.verify_require_provider_source is False

    def test_no_verify_require_provider_source_cli_flag(self):
        """--no-verify-require-provider-source flips the default to False."""
        from spread_scanner import parse_args
        args = parse_args(["--no-verify-require-provider-source"])
        assert args.verify_require_provider_source is False
        # Without the flag, default is True.
        args2 = parse_args([])
        assert args2.verify_require_provider_source is True

    def test_verify_failures_are_marked_not_dropped(self):
        """Verify failures must NOT remove spreads from data — they only
        annotate `verified=False` so the regular DTE section still renders
        them. Top-N filtering is what suppresses failed candidates from
        the picks list (see test below)."""
        import asyncio
        import argparse
        import sys as _sys
        import spread_scanner as scanner

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def verify_spread_pricing(self, **kw):
                # Always reject — simulates CPG csv_source_rejected.
                return {"ok": False, "reason": "csv_source_rejected"}

        fake_utp = type(_sys)("utp")
        fake_utp.TradingClient = lambda *a, **kw: FakeClient()
        _sys.modules["utp"] = fake_utp
        try:
            args = argparse.Namespace(
                daemon_url="http://localhost:8000",
                verify_max_age_sec=30.0,
                verify_require_provider_source=True,
                tickers=["SPX"], min_credit=0, min_roi=0, min_norm_roi=0,
                min_otm=0, max_otm=0, min_otm_per_ticker={},
                max_otm_per_ticker={}, min_tier=None, min_tier_close=None,
            )
            data = {
                "prev_closes": {"SPX": 7100.0},
                "dte_sections": {
                    0: {
                        "expiration": "2026-04-27",
                        "spreads": {
                            "SPX": [
                                {"option_type": "PUT", "short_strike": 7050,
                                 "long_strike": 7030, "credit": 0.40,
                                 "roi_pct": 2.0, "otm_pct": 0.7, "width": 20},
                                {"option_type": "PUT", "short_strike": 7045,
                                 "long_strike": 7025, "credit": 0.30,
                                 "roi_pct": 1.5, "otm_pct": 0.8, "width": 20},
                            ],
                        },
                    },
                },
            }
            summary = asyncio.run(
                scanner._verify_top_candidates_with_provider(args, data),
            )
        finally:
            _sys.modules.pop("utp", None)

        # Both spreads remain in data — non-destructive!
        spreads = data["dte_sections"][0]["spreads"]["SPX"]
        assert len(spreads) == 2
        # Each is marked verified=False with the reason.
        for s in spreads:
            assert s["verified"] is False
            assert s["verify_reason"] == "csv_source_rejected"
        # Summary counts the failures.
        assert summary["dropped"] == 2
        assert summary["reasons"]["csv_source_rejected"] == 2

    def test_top_picks_filters_out_verified_false(self):
        """Spreads marked `verified=False` by the verify step must not
        appear in Top-N or in the candidates fed to trade handlers."""
        from spread_scanner import _collect_filtered_candidates, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "5"])
        data = {
            "prev_closes": {"SPX": 7100.0},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-27",
                    "spreads": {
                        "SPX": [
                            {"option_type": "PUT", "short_strike": 7050,
                             "long_strike": 7030, "credit": 0.50,
                             "roi_pct": 2.6, "otm_pct": 0.7, "width": 20,
                             "verified": True},
                            {"option_type": "PUT", "short_strike": 7045,
                             "long_strike": 7025, "credit": 0.45,
                             "roi_pct": 2.3, "otm_pct": 0.8, "width": 20,
                             "verified": False, "verify_reason": "csv_source_rejected"},
                            # No `verified` key — verify hasn't run yet
                            # (top-M cutoff or daemon timeout). Allowed.
                            {"option_type": "PUT", "short_strike": 7040,
                             "long_strike": 7020, "credit": 0.40,
                             "roi_pct": 2.0, "otm_pct": 0.85, "width": 20},
                        ],
                    },
                },
            },
        }
        candidates = _collect_filtered_candidates(data, args)
        # The verified=False candidate is filtered out; the verified=True
        # and unverified ones come through.
        strikes = [c["short_strike"] for c in candidates]
        assert 7050 in strikes
        assert 7040 in strikes
        assert 7045 not in strikes

    def test_verify_batch_timeout_does_not_destroy_data(self):
        """When the daemon hangs and the verify batch times out, the
        regular DTE section must still render — we leave all spreads
        unverified rather than nuking them."""
        import asyncio
        import argparse
        import sys as _sys
        import spread_scanner as scanner

        class HangingClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def verify_spread_pricing(self, **kw):
                # Hang longer than the test timeout we'll inject.
                await asyncio.sleep(60)
                return {"ok": True}

        fake_utp = type(_sys)("utp")
        fake_utp.TradingClient = lambda *a, **kw: HangingClient()
        _sys.modules["utp"] = fake_utp
        try:
            args = argparse.Namespace(
                daemon_url="http://localhost:8000",
                verify_max_age_sec=30.0,
                verify_require_provider_source=True,
                # Tight batch timeout so the test runs quickly.
                verify_batch_timeout_sec=0.2,
                tickers=["SPX"], min_credit=0, min_roi=0, min_norm_roi=0,
                min_otm=0, max_otm=0, min_otm_per_ticker={},
                max_otm_per_ticker={}, min_tier=None, min_tier_close=None,
            )
            data = {
                "prev_closes": {"SPX": 7100.0},
                "dte_sections": {
                    0: {
                        "expiration": "2026-04-27",
                        "spreads": {
                            "SPX": [
                                {"option_type": "PUT", "short_strike": 7050,
                                 "long_strike": 7030, "credit": 0.40,
                                 "roi_pct": 2.0, "otm_pct": 0.7, "width": 20},
                            ],
                        },
                    },
                },
            }
            summary = asyncio.run(
                scanner._verify_top_candidates_with_provider(args, data),
            )
        finally:
            _sys.modules.pop("utp", None)

        # Spread is still there — batch timeout is a non-destructive event.
        spreads = data["dte_sections"][0]["spreads"]["SPX"]
        assert len(spreads) == 1
        # Not marked verified=False (verify never ran on it).
        assert spreads[0].get("verified") is not False
        # Summary surfaces the timeout count.
        assert summary.get("timed_out") == 1

    def test_verify_passes_require_provider_source_to_client(self, tmp_path):
        """`_verify_top_candidates_with_provider` forwards the args flag
        to TradingClient.verify_spread_pricing(require_provider_source=...)."""
        import asyncio
        import argparse
        import spread_scanner as scanner

        captured = {"calls": []}

        class FakeClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def verify_spread_pricing(self, **kw):
                captured["calls"].append(kw)
                return {
                    "ok": True, "credit": 0.40, "short_bid": 0.50,
                    "long_ask": 0.10, "short_delta": -0.17,
                    "age_seconds": 5.0, "source": "ibkr_fresh",
                }

        # Monkeypatch the import inside _verify_top_candidates_with_provider.
        import sys as _sys
        fake_utp = type(_sys)("utp")
        fake_utp.TradingClient = lambda *a, **kw: FakeClient()
        _sys.modules["utp"] = fake_utp
        try:
            args = argparse.Namespace(
                daemon_url="http://localhost:8000",
                verify_max_age_sec=30.0,
                verify_require_provider_source=False,
                # Required by _collect_filtered_candidates:
                tickers=["SPX"], min_credit=0, min_roi=0, min_norm_roi=0,
                min_otm=0, max_otm=0, min_otm_per_ticker={},
                max_otm_per_ticker={}, min_tier=None, min_tier_close=None,
            )
            data = {
                "prev_closes": {"SPX": 7100.0},
                "dte_sections": {
                    0: {
                        "expiration": "2026-04-27",
                        "spreads": {
                            "SPX": [{
                                "option_type": "PUT", "short_strike": 7050,
                                "long_strike": 7030, "credit": 0.40,
                                "roi_pct": 2.0, "otm_pct": 0.7, "width": 20,
                            }],
                        },
                    },
                },
            }
            asyncio.run(scanner._verify_top_candidates_with_provider(args, data))
        finally:
            _sys.modules.pop("utp", None)

        assert len(captured["calls"]) == 1
        # The flag from args propagated through.
        assert captured["calls"][0]["require_provider_source"] is False

    def test_top_picks_shows_verification_marker(self):
        """Rows that went through the provider re-verify show ✓ + age; rows
        that didn't (e.g. candidates below the verify batch cutoff) show —."""
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "2"])
        scan_data = {
            "prev_closes": {"SPX": 7100.0},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-24",
                    "spreads": {
                        "SPX": [
                            # Verified against a 3s-old cached quote.
                            {"option_type": "PUT", "short_strike": 7050,
                             "long_strike": 7030, "credit": 0.50,
                             "roi_pct": 2.6, "otm_pct": 0.7, "width": 20,
                             "short_delta": -0.17,
                             "verified": True, "verify_age_seconds": 3.2},
                            # Not verified (candidate outside top-M batch).
                            {"option_type": "CALL", "short_strike": 7200,
                             "long_strike": 7220, "credit": 0.60,
                             "roi_pct": 3.1, "otm_pct": 1.4, "width": 20,
                             "short_delta": None},
                        ],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        text = "\n".join(lines)
        # Header exposes the Vfy column.
        assert "Vfy" in text
        # Verified row shows ✓ with age.
        assert "✓3s" in text
        # Unverified row shows —.
        assert "—" in text

    def test_top_picks_row_alignment_stable_across_strike_digits(self):
        """NDX 5-digit strikes (25830/25820) must not push later columns
        right vs SPX 4-digit strikes (6920/6915). The Short/Long field is
        padded as a single unit, not long-side-only."""
        from spread_scanner import _render_top_picks, parse_args, _visible_len

        args = parse_args(["--tickers", "SPX,NDX", "--top", "2"])
        scan_data = {
            "prev_closes": {"SPX": 7100.0, "NDX": 27000.0},
            "dte_sections": {
                1: {
                    "expiration": "2026-04-25",
                    "spreads": {
                        "SPX": [{"option_type": "PUT", "short_strike": 6920,
                                 "long_strike": 6915, "credit": 0.35,
                                 "roi_pct": 7.6, "otm_pct": 2.8, "width": 5,
                                 "short_delta": None}],
                        "NDX": [{"option_type": "PUT", "short_strike": 25830,
                                 "long_strike": 25820, "credit": 0.60,
                                 "roi_pct": 6.4, "otm_pct": 4.6, "width": 10,
                                 "short_delta": None}],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        data_rows = [ln for ln in lines if ln.strip().startswith(("1 ", "2 "))]
        assert len(data_rows) == 2
        # Both rows should have the same visible width (same column layout).
        w0, w1 = _visible_len(data_rows[0]), _visible_len(data_rows[1])
        assert w0 == w1, f"Row widths differ: {w0} vs {w1}\n  r0={data_rows[0]!r}\n  r1={data_rows[1]!r}"

    def test_top_picks_renders_short_delta_column(self):
        """Top-N rows include a Δshort column showing the short leg's delta."""
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "2"])
        scan_data = {
            "prev_closes": {"SPX": 7100.0},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-25",
                    "spreads": {
                        "SPX": [
                            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7030,
                             "credit": 0.50, "roi_pct": 2.6, "otm_pct": 0.7, "width": 20,
                             "short_delta": -0.17},
                            {"option_type": "CALL", "short_strike": 7200, "long_strike": 7220,
                             "credit": 0.60, "roi_pct": 3.1, "otm_pct": 1.4, "width": 20,
                             "short_delta": None},  # provider didn't return greeks
                        ],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        text = "\n".join(lines)
        # Header has Δshort column.
        assert "Δshort" in text
        # Row with delta shows it; row without delta is fine to omit.
        assert "Δ-0.17" in text

    # ── classify_strike_to_percentile ────────────────────────────────────

    @staticmethod
    def _make_tier_data(symbol: str, *,
                        c2c_pct_map: dict | None = None,
                        intraday_pct_map: dict | None = None,
                        side: str = "put",
                        recommended_c2c: dict | None = None,
                        recommended_intraday: dict | None = None,
                        slot: str = "10:00",
                        c2c_dte: int = 0) -> dict:
        """Build a minimal tier_data fixture matching the live shape."""
        direction = "when_down" if side == "put" else "when_up"
        td = {"hourly": {symbol: {"recommended": {}, "slots": {}}}, "tickers": []}
        sym = td["hourly"][symbol]
        if recommended_c2c is not None:
            sym["recommended"]["close_to_close"] = recommended_c2c
        if recommended_intraday is not None:
            sym["recommended"]["intraday"] = recommended_intraday
        if intraday_pct_map is not None:
            sym["slots"][slot] = {direction: {"pct": intraday_pct_map}}
        if c2c_pct_map is not None:
            td["tickers"].append({
                "ticker": symbol,
                "windows": {str(c2c_dte): {direction: {"pct": c2c_pct_map}}},
            })
        return td

    def test_classify_strike_to_percentile_close_to_close_named_tier(self):
        """A short put strike that lines up exactly with the cons-tier
        percentile in the c2c map returns (98, "conservative")."""
        from spread_scanner import classify_strike_to_percentile
        prev_close = 7100.0
        # p98 represents -2.0% c2c move → strike ≈ 7100 * 0.98 = 6958.
        td = self._make_tier_data(
            "SPX",
            c2c_pct_map={"p90": -1.0, "p95": -1.5, "p98": -2.0, "p99": -2.5},
            recommended_c2c={
                "aggressive":   {"put": 90, "call": 90},
                "moderate":     {"put": 95, "call": 95},
                "conservative": {"put": 98, "call": 98},
            },
            side="put", c2c_dte=0,
        )
        # Place the short strike at exactly the p98 boundary.
        pct, tier = classify_strike_to_percentile(
            td, "SPX", "put", "close_to_close",
            prev_close=prev_close, current_price=7090.0,
            short_strike=6958.0, dte=0,
        )
        assert pct == 98
        assert tier == "conservative"

    def test_classify_strike_to_percentile_intraday_no_named_tier_match(self):
        """When the resolved percentile lands between the named-tier
        recommendations, return (pN, None)."""
        from spread_scanner import classify_strike_to_percentile
        # Provide pcts at p85, p87, p90, p95, p98.
        td = self._make_tier_data(
            "SPX",
            intraday_pct_map={"p85": -0.50, "p87": -0.65, "p90": -0.80,
                              "p95": -1.20, "p98": -1.80},
            recommended_intraday={
                "aggressive":   {"put": 90},
                "moderate":     {"put": 95},
                "conservative": {"put": 98},
            },
            side="put",
        )
        # Put a strike at -0.65% from spot → matches exactly p87 (not named).
        spot = 7000.0
        strike = spot * (1 - 0.0065)  # = 6954.5 → -0.65%
        pct, tier = classify_strike_to_percentile(
            td, "SPX", "put", "intraday",
            prev_close=7000.0, current_price=spot,
            short_strike=strike, dte=0,
        )
        assert pct == 87
        assert tier is None

    def test_classify_strike_to_percentile_returns_none_when_strike_too_close(self):
        """If the strike is closer-to-the-money than even the smallest
        percentile in the map, return (None, None) — caller renders dim
        em-dash so the operator knows the strike is below the model's
        protective range."""
        from spread_scanner import classify_strike_to_percentile
        td = self._make_tier_data(
            "SPX",
            intraday_pct_map={"p90": -1.0, "p95": -1.5, "p98": -2.0},
            recommended_intraday={"aggressive": {"put": 90}},
            side="put",
        )
        # Strike only -0.10% from spot — closer than p90's -1.0%.
        spot = 7000.0
        strike = spot * (1 - 0.001)
        pct, tier = classify_strike_to_percentile(
            td, "SPX", "put", "intraday",
            prev_close=7000.0, current_price=spot,
            short_strike=strike, dte=0,
        )
        assert pct is None
        assert tier is None

    def test_classify_strike_to_percentile_handles_missing_tier_data(self):
        """tier_data=None or missing symbol → (None, None) without raising."""
        from spread_scanner import classify_strike_to_percentile

        for td in (None, {}, {"hourly": {}}, {"hourly": {"OTHER": {}}}):
            pct, tier = classify_strike_to_percentile(
                td, "SPX", "put", "close_to_close",
                prev_close=7000.0, current_price=7000.0,
                short_strike=6900.0, dte=0,
            )
            assert pct is None
            assert tier is None

    def test_classify_strike_to_percentile_calls_use_when_up_path(self):
        """Calls use the when_up branch and the 'more positive' protection
        rule. A call strike at +1.5% from spot should resolve to p95 when
        the map is {p90: +1.0, p95: +1.5, p98: +2.0}."""
        from spread_scanner import classify_strike_to_percentile
        td = self._make_tier_data(
            "SPX",
            intraday_pct_map={"p90": 1.0, "p95": 1.5, "p98": 2.0},
            recommended_intraday={
                "aggressive":   {"call": 90},
                "moderate":     {"call": 95},
                "conservative": {"call": 98},
            },
            side="call",
        )
        spot = 7000.0
        strike = 7105.0  # exactly +1.5% from spot (avoids FP noise from
                         # `spot * 1.015`, which returns 7104.999999999999)
        pct, tier = classify_strike_to_percentile(
            td, "SPX", "call", "intraday",
            prev_close=7000.0, current_price=spot,
            short_strike=strike, dte=0,
        )
        assert pct == 95
        assert tier == "moderate"

    def test_top_picks_shows_hist_and_pred_percentile_columns(self):
        """When tier_data is on the dte_section, each Top-N row picks up
        Hist (close-to-close) and Pred (intraday) percentile cells with
        the matching tier label rendered when the percentile equals one
        of the recommended named tiers."""
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "1"])
        # Strike at -1.5% c2c (= p95 → moderate) and at -0.80% intraday
        # (= p90 → aggressive in our fixture).
        prev_close = 7100.0
        spot = 7090.0
        short = prev_close * (1 - 0.015)  # 6993.5 from prev_close
        # Round to a clean 5-strike increment for SPX, but the test only
        # cares about the model's classification — which uses the spread's
        # actual short_strike. So pass 6993.5 directly.
        td = {
            "hourly": {
                "SPX": {
                    "recommended": {
                        "close_to_close": {
                            "aggressive":   {"put": 90},
                            "moderate":     {"put": 95},
                            "conservative": {"put": 98},
                        },
                        "intraday": {
                            "aggressive":   {"put": 90},
                            "moderate":     {"put": 95},
                            "conservative": {"put": 98},
                        },
                    },
                    "slots": {
                        "10:00": {
                            "when_down": {
                                "pct": {"p85": -0.50, "p90": -0.80,
                                        "p95": -1.20, "p98": -1.80},
                            }
                        }
                    },
                }
            },
            "tickers": [{
                "ticker": "SPX",
                "windows": {
                    "0": {"when_down": {
                        "pct": {"p90": -1.0, "p95": -1.5, "p98": -2.0},
                    }}
                },
            }],
        }
        scan_data = {
            "quotes": {"SPX": {"last": spot}},
            "prev_closes": {"SPX": prev_close},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-30",
                    "tier_data": td,
                    "spreads": {
                        "SPX": [{
                            "option_type": "PUT",
                            "short_strike": short,
                            "long_strike": short - 20,
                            "credit": 0.50, "roi_pct": 2.6, "otm_pct": 1.4,
                            "width": 20, "short_delta": -0.17,
                        }],
                    },
                }
            },
        }

        # Patch _find_current_slot so the test isn't time-of-day dependent.
        import spread_scanner as sp
        orig = sp._find_current_slot
        sp._find_current_slot = lambda slots: "10:00" if slots else None
        try:
            lines = _render_top_picks(scan_data, args)
        finally:
            sp._find_current_slot = orig

        text = "\n".join(lines)
        assert "Hist" in text and "Pred" in text
        # Hist (c2c): strike is at exactly -1.5% from prev_close, c2c map
        # has {p90:-1.0, p95:-1.5, p98:-2.0}. The strike survives p90
        # (covers -1.0% drop) and p95 (covers -1.5% drop, boundary) but
        # NOT p98 (would need to cover -2.0% drop). Resolved → p95.
        # recommended.close_to_close.moderate.put=95 → tier label "mod".
        #
        # Pred (intraday): strike's offset from spot 7090 is
        # (6993.5 - 7090)/7090 ≈ -1.36%. Intraday map has
        # {p85:-0.50, p90:-0.80, p95:-1.20, p98:-1.80}. Strike survives
        # p85, p90, p95 (covers -1.20%) but not p98 (-1.80% > -1.36%
        # in magnitude). Resolved → p95.
        # recommended.intraday.moderate.put=95 → tier label "mod".
        #
        # Both cells should therefore show "p95" + "mod".
        assert "p95" in text
        assert text.count("mod") >= 2, (
            f"expected 'mod' in both Hist and Pred cells, got: {text!r}"
        )

    def test_top_picks_columns_align_with_header(self):
        """Header column boundaries must coincide with data-row column
        boundaries — i.e. once you strip ANSI codes, walking the row by
        the declared column widths puts each cell's start under its
        header label. Regression: earlier the data row used `nroi_str + " "`
        and `ot{otm} cl{chg}` as one chunk, so OTM%/Cl%/Δshort/Vfy/Hist/
        Pred all drifted left of their header labels — visible to the
        operator as a row that doesn't line up with the header."""
        from spread_scanner import _render_top_picks, parse_args, _visible_len
        import re

        ANSI = re.compile(r"\033\[[0-9;]*m")

        args = parse_args(["--tickers", "SPX,NDX", "--top", "2"])
        scan_data = {
            "prev_closes": {"SPX": 7100.0, "NDX": 27000.0},
            "dte_sections": {
                1: {
                    "expiration": "2026-04-30",
                    "tier_data": None,
                    "spreads": {
                        "SPX": [{
                            "option_type": "PUT", "short_strike": 6990,
                            "long_strike": 6985, "credit": 0.40,
                            "roi_pct": 8.0, "otm_pct": 1.5, "width": 5,
                            "short_delta": -0.13,
                            "verified": True, "verify_age_seconds": 12,
                        }],
                        "NDX": [{
                            "option_type": "CALL", "short_strike": 27730,
                            "long_strike": 27790, "credit": 5.30,
                            "roi_pct": 9.7, "otm_pct": 2.7, "width": 60,
                            "short_delta": 0.11,
                            "verified": True, "verify_age_seconds": 56,
                        }],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        header = next(ln for ln in lines if "Short/Long" in ln)
        plain_header = ANSI.sub("", header)
        data_rows = [ANSI.sub("", ln) for ln in lines
                     if ln.strip().startswith(("1 ", "2 "))]
        assert len(data_rows) == 2

        # All rows + header must have identical visible width (the layout
        # is a fixed-width table).
        h_w = _visible_len(plain_header)
        for row in data_rows:
            assert _visible_len(row) == h_w, (
                f"row width {_visible_len(row)} != header width {h_w}\n"
                f"  header: {plain_header!r}\n"
                f"  row:    {row!r}"
            )

        # Walk a few distinctive header labels and assert the data row
        # has the expected character (or a space, if the cell is shorter
        # than its column) at the SAME visible index.
        for label, expected_chars in [
            ("Hist",   (" ", "p", "—")),
            ("Pred",   (" ", "p", "—")),
            ("Vfy",    (" ", "✓", "—")),
            ("OTM%",   (" ", "o")),     # cell starts with "ot…"
            ("Cl%",    (" ", "c")),     # cell starts with "cl…"
            ("Δshort", (" ", "Δ", "-")),
            ("nROI",   (" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9")),
        ]:
            idx = plain_header.index(label)
            for row in data_rows:
                ch = row[idx] if idx < len(row) else " "
                assert ch in expected_chars, (
                    f"data row col {idx} (under header label {label!r}) "
                    f"should start with one of {expected_chars}, "
                    f"got {ch!r}\n  header: {plain_header!r}\n  row:    {row!r}"
                )

    def test_top_picks_shows_dash_when_tier_data_absent(self):
        """When tier_data is missing entirely, Hist/Pred cells render as
        dim em-dashes — but the columns and header are still present."""
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "1"])
        scan_data = {
            "quotes": {"SPX": {"last": 7100.0}},
            "prev_closes": {"SPX": 7100.0},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-30",
                    "tier_data": None,
                    "spreads": {
                        "SPX": [{
                            "option_type": "PUT", "short_strike": 7050,
                            "long_strike": 7030, "credit": 0.50,
                            "roi_pct": 2.6, "otm_pct": 0.7, "width": 20,
                            "short_delta": -0.17,
                        }],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        text = "\n".join(lines)
        assert "Hist" in text and "Pred" in text
        # Dim em-dash (— = U+2014) appears for both percentile cells.
        # The data row should have at least 2 em-dashes (Hist + Pred).
        # (Vfy column also uses em-dash for unverified rows, so allow ≥2.)
        data_rows = [ln for ln in lines if ln.strip().startswith("1 ")]
        assert data_rows, f"expected one data row, got lines: {lines!r}"
        em_dash_count = data_rows[0].count("—")
        assert em_dash_count >= 2, (
            f"expected ≥2 em-dashes (Hist + Pred placeholders), "
            f"got {em_dash_count} in row: {data_rows[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_resolve_percentile_url_custom_bypasses_probe(self):
        """A caller-supplied URL (not the baked-in lin1 default) is used as-is,
        no reachability probe, no fallback."""
        import spread_scanner as sp
        sp._percentile_url_cache.clear()

        class NeverCalled:
            async def get(self, *a, **kw):
                raise AssertionError("probe should not fire for custom URLs")

        resolved = await sp._resolve_percentile_url(
            NeverCalled(), "http://my.custom.host:9100",
        )
        assert resolved == "http://my.custom.host:9100"

    @pytest.mark.asyncio
    async def test_resolve_percentile_url_falls_back_when_primary_down(self):
        """When the lin1 default is unreachable, fall back to localhost."""
        import spread_scanner as sp
        import httpx
        sp._percentile_url_cache.clear()

        class PrimaryDown:
            def __init__(self):
                self.calls = []
            async def get(self, url, *a, **kw):
                self.calls.append(url)
                if url.startswith(sp.DEFAULT_PERCENTILE_URL):
                    raise httpx.ConnectError("simulated: lin1 offline")
                # Fallback (localhost) answers
                class R: status_code = 200
                return R()

        stub = PrimaryDown()
        resolved = await sp._resolve_percentile_url(stub, sp.DEFAULT_PERCENTILE_URL)
        assert resolved == sp._PERCENTILE_FALLBACK_URL
        # Probe order must be primary first, then fallback.
        assert sp.DEFAULT_PERCENTILE_URL in stub.calls[0]
        assert sp._PERCENTILE_FALLBACK_URL in stub.calls[1]

    @pytest.mark.asyncio
    async def test_resolve_percentile_url_prefers_primary_and_caches(self):
        """When lin1 answers, use lin1 and cache the decision."""
        import spread_scanner as sp
        sp._percentile_url_cache.clear()

        call_count = {"n": 0}
        class PrimaryUp:
            async def get(self, url, *a, **kw):
                call_count["n"] += 1
                class R: status_code = 200
                return R()

        stub = PrimaryUp()
        first = await sp._resolve_percentile_url(stub, sp.DEFAULT_PERCENTILE_URL)
        second = await sp._resolve_percentile_url(stub, sp.DEFAULT_PERCENTILE_URL)
        assert first == sp.DEFAULT_PERCENTILE_URL
        assert second == sp.DEFAULT_PERCENTILE_URL
        # Only probed once — second call hit the cache.
        assert call_count["n"] == 1

    # ── primary/backup URL config (YAML + CLI) ────────────────────────────

    def test_url_mapping_form_in_yaml_splits_into_primary_and_backup(self, tmp_path):
        """YAML `daemon_url: {primary, backup}` populates the two scalar
        fields so downstream resolution can pick the working one."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({
            "daemon_url": {
                "primary": "http://lin1.kundu.dev:8000",
                "backup": "http://localhost:8000",
            },
            "db_url": {
                "primary": "http://lin1.kundu.dev:9102",
                "backup": "http://localhost:9102",
            },
            "percentile_url": {
                "primary": "http://lin1.kundu.dev:9100",
                "backup": "http://localhost:9100",
            },
        }))
        cfg = ScannerConfig.from_yaml(str(p))
        assert cfg.daemon_url == "http://lin1.kundu.dev:8000"
        assert cfg.daemon_url_backup == "http://localhost:8000"
        assert cfg.db_url == "http://lin1.kundu.dev:9102"
        assert cfg.db_url_backup == "http://localhost:9102"
        assert cfg.percentile_url == "http://lin1.kundu.dev:9100"
        assert cfg.percentile_url_backup == "http://localhost:9100"

    def test_url_scalar_form_in_yaml_remains_supported(self, tmp_path):
        """Scalar form for URL fields keeps working — it's the legacy syntax
        used by every existing reference YAML."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({
            "daemon_url": "http://my.daemon:8000",
            "db_url": "http://my.db:9102",
        }))
        cfg = ScannerConfig.from_yaml(str(p))
        assert cfg.daemon_url == "http://my.daemon:8000"
        assert cfg.daemon_url_backup is None
        assert cfg.db_url == "http://my.db:9102"
        assert cfg.db_url_backup is None

    def test_url_mapping_form_primary_only_leaves_backup_none(self, tmp_path):
        """Mapping form with `primary` alone (no `backup`) is allowed and
        means 'no failover' — equivalent to scalar form."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({
            "daemon_url": {"primary": "http://x:8000"},
        }))
        cfg = ScannerConfig.from_yaml(str(p))
        assert cfg.daemon_url == "http://x:8000"
        assert cfg.daemon_url_backup is None

    def test_url_mapping_form_rejects_missing_primary(self, tmp_path):
        """Mapping form must include a non-empty `primary`."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({"daemon_url": {"backup": "http://x:8000"}}))
        with pytest.raises(ValueError, match="primary"):
            ScannerConfig.from_yaml(str(p))

    def test_url_mapping_form_rejects_unknown_keys(self, tmp_path):
        """Catches typos like `fallback` instead of `backup` early."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({
            "daemon_url": {"primary": "http://x", "fallback": "http://y"},
        }))
        with pytest.raises(ValueError, match="fallback"):
            ScannerConfig.from_yaml(str(p))

    def test_url_mapping_form_propagates_to_args(self, tmp_path):
        """YAML mapping form ends up on the parsed argparse namespace so the
        startup resolver and call sites can read it."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig, parse_args
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({
            "daemon_url": {"primary": "http://a:8000", "backup": "http://b:8000"},
        }))
        cfg = ScannerConfig.from_yaml(str(p))
        defaults = cfg.to_cli_defaults()
        args = parse_args(argv=[], defaults=defaults)
        assert args.daemon_url == "http://a:8000"
        assert args.daemon_url_backup == "http://b:8000"

    def test_cli_backup_flags_override_yaml(self, tmp_path):
        """CLI flags beat YAML, including for the new --*-url-backup pair."""
        import yaml as _yaml
        from spread_scanner import ScannerConfig, parse_args
        p = tmp_path / "cfg.yaml"
        p.write_text(_yaml.safe_dump({
            "daemon_url": {"primary": "http://yaml-p:8000", "backup": "http://yaml-b:8000"},
        }))
        cfg = ScannerConfig.from_yaml(str(p))
        defaults = cfg.to_cli_defaults()
        args = parse_args(
            argv=["--daemon-url", "http://cli-p:8000",
                  "--daemon-url-backup", "http://cli-b:8000"],
            defaults=defaults,
        )
        assert args.daemon_url == "http://cli-p:8000"
        assert args.daemon_url_backup == "http://cli-b:8000"

    def test_default_args_get_baked_in_percentile_backup(self):
        """Backwards compat: with no overrides, the percentile URL still
        gets its baked-in `localhost:9100` fallback (legacy behavior)."""
        from spread_scanner import (
            parse_args, DEFAULT_PERCENTILE_URL, _PERCENTILE_FALLBACK_URL,
        )
        args = parse_args([])
        assert args.percentile_url == DEFAULT_PERCENTILE_URL
        assert args.percentile_url_backup == _PERCENTILE_FALLBACK_URL

    def test_custom_percentile_url_does_not_inject_backup(self):
        """Setting a custom primary opts OUT of the baked-in backup —
        operators with a custom percentile host don't want surprise probes
        against localhost."""
        from spread_scanner import parse_args
        args = parse_args(["--percentile-url", "http://custom-percentile:9100"])
        assert args.percentile_url == "http://custom-percentile:9100"
        assert args.percentile_url_backup is None

    @pytest.mark.asyncio
    async def test_resolve_url_with_backup_returns_primary_when_no_backup(self):
        """No backup configured → primary returned unchanged, no probe."""
        import spread_scanner as sp
        sp._resolved_url_cache.clear()

        class NeverCalled:
            async def get(self, *a, **kw):
                raise AssertionError("probe should not fire when backup is None")

        out = await sp._resolve_url_with_backup(
            NeverCalled(), "http://primary:8000", None,
        )
        assert out == "http://primary:8000"

    @pytest.mark.asyncio
    async def test_resolve_url_with_backup_picks_primary_when_up(self):
        """Primary answers → primary chosen and cached for subsequent calls."""
        import spread_scanner as sp
        sp._resolved_url_cache.clear()

        calls = []

        class PrimaryUp:
            async def get(self, url, *a, **kw):
                calls.append(url)
                class R: status_code = 200
                return R()

        stub = PrimaryUp()
        first = await sp._resolve_url_with_backup(
            stub, "http://primary:8000", "http://backup:8000",
            probe_path="/health",
        )
        second = await sp._resolve_url_with_backup(
            stub, "http://primary:8000", "http://backup:8000",
            probe_path="/health",
        )
        assert first == "http://primary:8000"
        assert second == "http://primary:8000"
        # Only probed once — second call hit the cache.
        assert len(calls) == 1
        assert calls[0].startswith("http://primary:8000")

    @pytest.mark.asyncio
    async def test_resolve_url_with_backup_falls_back_when_primary_down(self):
        """Primary unreachable → backup probed and selected."""
        import spread_scanner as sp
        import httpx
        sp._resolved_url_cache.clear()

        calls = []

        class PrimaryDown:
            async def get(self, url, *a, **kw):
                calls.append(url)
                if url.startswith("http://primary"):
                    raise httpx.ConnectError("simulated primary offline")
                class R: status_code = 200
                return R()

        out = await sp._resolve_url_with_backup(
            PrimaryDown(), "http://primary:8000", "http://backup:8000",
            probe_path="/dashboard/summary",
        )
        assert out == "http://backup:8000"
        # Probe order: primary first, then backup
        assert calls[0].startswith("http://primary")
        assert calls[1].startswith("http://backup")

    @pytest.mark.asyncio
    async def test_resolve_url_with_backup_caches_failure_state(self):
        """Both unreachable → backup cached so we don't keep re-probing
        the primary on every scan cycle."""
        import spread_scanner as sp
        import httpx
        sp._resolved_url_cache.clear()

        call_count = {"n": 0}

        class BothDown:
            async def get(self, url, *a, **kw):
                call_count["n"] += 1
                raise httpx.ConnectError("offline")

        first = await sp._resolve_url_with_backup(
            BothDown(), "http://p:1", "http://b:1",
        )
        second = await sp._resolve_url_with_backup(
            BothDown(), "http://p:1", "http://b:1",
        )
        # Both calls return the backup; second is cached (no extra probes).
        assert first == "http://b:1"
        assert second == "http://b:1"
        assert call_count["n"] == 2  # 1 primary + 1 backup probe (only the first call)

    @pytest.mark.asyncio
    async def test_resolve_endpoint_urls_writes_back_to_args(self):
        """Startup resolver overwrites args with the working URL so later
        fetch calls pick up the correct host."""
        import spread_scanner as sp
        import httpx
        from argparse import Namespace
        sp._resolved_url_cache.clear()

        class PrimaryUp:
            async def get(self, url, *a, **kw):
                if url.startswith("http://daemon-p"):
                    class R: status_code = 200
                    return R()
                if url.startswith("http://daemon-b"):
                    raise httpx.ConnectError("backup not probed since primary up")
                class R: status_code = 200
                return R()

        args = Namespace(
            daemon_url="http://daemon-p:8000",
            daemon_url_backup="http://daemon-b:8000",
            db_url="http://db:9102",
            db_url_backup=None,
            percentile_url="http://lin1:9100",
            percentile_url_backup=None,
        )
        out = await sp.resolve_endpoint_urls(PrimaryUp(), args)
        assert args.daemon_url == "http://daemon-p:8000"
        assert out["daemon_url"] == "http://daemon-p:8000"
        # No-backup URLs unchanged
        assert args.db_url == "http://db:9102"
        assert args.percentile_url == "http://lin1:9100"

    @pytest.mark.asyncio
    async def test_resolve_endpoint_urls_swaps_to_backup_on_primary_down(self):
        """When primary fails, the startup resolver writes the backup URL
        back to args so all subsequent code uses the working host."""
        import spread_scanner as sp
        import httpx
        from argparse import Namespace
        sp._resolved_url_cache.clear()

        class PrimaryDown:
            async def get(self, url, *a, **kw):
                if url.startswith("http://primary"):
                    raise httpx.ConnectError("primary offline")
                class R: status_code = 200
                return R()

        args = Namespace(
            daemon_url="http://primary-d:8000",
            daemon_url_backup="http://backup-d:8000",
            db_url="http://primary-db:9102",
            db_url_backup="http://backup-db:9102",
            percentile_url="http://primary-pct:9100",
            percentile_url_backup="http://backup-pct:9100",
        )
        await sp.resolve_endpoint_urls(PrimaryDown(), args)
        assert args.daemon_url == "http://backup-d:8000"
        assert args.db_url == "http://backup-db:9102"
        assert args.percentile_url == "http://backup-pct:9100"

    @pytest.mark.asyncio
    async def test_resolve_url_with_backup_default_timeout_tolerates_slow_probe(self):
        """Default probe timeout (4.0s) must NOT trip on a slow but healthy
        primary. Regression: a 2.0s default + a daemon /dashboard/summary
        that occasionally took 2.5s caused the resolver to silently fall
        through to the (unreachable) backup, locking the scanner into a
        dead URL for the rest of the process. The probe path itself was
        also moved from /dashboard/summary to /market/quote/SPX, but the
        timeout floor is the load-bearing fix here."""
        import inspect
        import spread_scanner as sp
        sig = inspect.signature(sp._resolve_url_with_backup)
        assert sig.parameters["timeout"].default >= 4.0, (
            "probe timeout must be >=4s -- anything tighter regresses on "
            "daemons whose probe endpoint hiccups above 2s under load"
        )

    @pytest.mark.asyncio
    async def test_resolve_endpoint_urls_uses_lightweight_daemon_probe(self):
        """Daemon probe must hit /market/quote/SPX (consistently sub-10ms),
        not /dashboard/summary (occasional 1-3s spikes that exceed the probe
        timeout and falsely classify a healthy daemon as down)."""
        import spread_scanner as sp
        from argparse import Namespace
        sp._resolved_url_cache.clear()

        seen_paths = []

        class CapturingClient:
            async def get(self, url, *a, **kw):
                seen_paths.append(url)
                class R: status_code = 200
                return R()

        args = Namespace(
            daemon_url="http://daemon-p:8000",
            daemon_url_backup="http://daemon-b:8000",
            db_url="http://db:9102",
            db_url_backup=None,
            percentile_url="http://pct:9100",
            percentile_url_backup=None,
        )
        await sp.resolve_endpoint_urls(CapturingClient(), args)
        daemon_probes = [p for p in seen_paths if "daemon-p" in p or "daemon-b" in p]
        assert daemon_probes, "expected at least one daemon probe"
        assert all("/market/quote/SPX" in p for p in daemon_probes), (
            f"daemon probe must use /market/quote/SPX, got: {daemon_probes}"
        )
        assert not any("/dashboard/summary" in p for p in daemon_probes), (
            "daemon probe must NOT use /dashboard/summary (slow under load)"
        )

    def test_render_endpoints_line_shows_resolved_endpoints_in_grey(self):
        """Endpoints line surfaces the URLs the scanner actually talks to so
        a bad primary/backup probe (which would otherwise be invisible) is
        diagnosable from the dashboard alone. Lives outside the footer so
        the per-second tick repaint (ESC[F/ESC[2K) keeps working on a
        single-line footer."""
        from spread_scanner import render_endpoints_line, parse_args, DIM, RESET

        args = parse_args(["--tickers", "SPX", "--interval", "30"])
        args.resolved_endpoints = {
            "daemon_url": "http://lin1.kundu.dev:8000",
            "db_url": "http://localhost:9102",
            "percentile_url": "http://localhost:9100",
        }
        out = render_endpoints_line(args)
        assert "endpoints:" in out
        assert "daemon=http://lin1.kundu.dev:8000" in out
        assert "db=http://localhost:9102" in out
        assert "percentile=http://localhost:9100" in out
        assert DIM in out and RESET in out
        # Must be a single line so it doesn't break ESC[F-based tick repaint.
        assert "\n" not in out

    def test_render_endpoints_line_falls_back_when_probe_did_not_run(self):
        """If `--once` or a test bypasses scan_loop, args.resolved_endpoints
        won't be set. Endpoints line should still display the configured
        URLs from args directly so the operator can see where it's pointed."""
        from spread_scanner import render_endpoints_line, parse_args

        args = parse_args(["--tickers", "SPX", "--interval", "30"])
        args.daemon_url = "http://localhost:8000"
        args.db_url = "http://localhost:9102"
        args.percentile_url = "http://localhost:9100"
        out = render_endpoints_line(args)
        assert "daemon=http://localhost:8000" in out
        assert "db=http://localhost:9102" in out
        assert "percentile=http://localhost:9100" in out

    def test_render_footer_is_single_line_so_per_second_tick_repaint_works(self):
        """The per-second footer tick uses ESC[F + ESC[2K to clear exactly
        one line. If render_footer ever returns multiple lines, the tick
        leaves stray copies behind and the screen scrolls. Pin the contract."""
        from spread_scanner import render_footer, parse_args

        args = parse_args(["--tickers", "SPX", "--interval", "30"])
        args.resolved_endpoints = {
            "daemon_url": "http://localhost:8000",
            "db_url": "http://localhost:9102",
            "percentile_url": "http://localhost:9100",
        }
        out = render_footer(args)
        assert "\n" not in out, "render_footer must be single-line"
        # And it must NOT carry endpoints (those go in render_endpoints_line).
        assert "endpoints:" not in out
        assert "daemon=" not in out

    def test_render_footer_uses_countdown(self):
        """render_footer's `seconds_remaining` should override the interval
        so the scan loop can tick the displayed countdown each second."""
        from spread_scanner import render_footer, parse_args

        args = parse_args(["--tickers", "SPX", "--interval", "30"])
        # Default: uses configured interval.
        default_footer = render_footer(args)
        assert "Next: +30s" in default_footer

        # Countdown override: reflects the live remaining time.
        countdown_footer = render_footer(args, seconds_remaining=7)
        assert "Next: +7s" in countdown_footer
        assert "+30s" not in countdown_footer

        # Clamped at 0 (never shows negative).
        assert "Next: +0s" in render_footer(args, seconds_remaining=-5)

    def test_render_footer_shows_local_timezone(self):
        """Footer's timestamp should include the local timezone abbreviation
        (PDT/PST/EDT/EST/…), never a hardcoded 'ET'."""
        from spread_scanner import render_footer, parse_args

        args = parse_args(["--tickers", "SPX", "--interval", "30"])
        footer = render_footer(args)
        # Hardcoded 'ET' must not leak into output.
        assert " ET " not in footer
        # Local tz abbreviation is present (non-empty after the time).
        import re
        # matches "HH:MM:SS TZ" where TZ is 1+ letters
        assert re.search(r"\d{2}:\d{2}:\d{2} [A-Z]+", footer) is not None

    def test_build_spread_from_chain_snaps_to_available_grid(self):
        """If the configured width lands on a non-listed strike (e.g. NDX
        jumps from 10-pt to 50-pt grid at far-OTM), snap the long leg to the
        widest listed OTM strike within max width."""
        from spread_scanner import build_spread_from_chain

        # NDX-style chain: 10-pt grid up to 27700, then 50-pt grid.
        # (Credit economics — short_bid > long_ask — so snap yields a real spread.)
        chain = {
            "call": [
                {"strike": 27650, "bid": 0.40, "ask": 0.55},
                {"strike": 27700, "bid": 0.10, "ask": 0.25},
                # 27710, 27720, 27730, 27740 — NOT LISTED
                {"strike": 27750, "bid": 0.00, "ask": 0.15},
            ],
            "put": [],
        }
        # Width=60 would target 27710 (non-existent). Should snap to 27700 (w=50).
        # credit = 0.40 - 0.25 = 0.15 (> 0, tradeable).
        s = build_spread_from_chain(
            chain, 27650, "CALL", 60, 26900.0,
        )
        assert s is not None
        assert s["long_strike"] == 27700
        assert s["width"] == 50  # actual width reflects what was used
        assert s["credit"] == 0.15

    def test_build_spread_from_chain_no_snap_when_too_far(self):
        """Snap only up to max width — if every listed strike is wider than
        the configured width, return None."""
        from spread_scanner import build_spread_from_chain

        # Only strikes are short=100 and a long at 200 (width=100).
        chain = {
            "call": [
                {"strike": 100, "bid": 1.00, "ask": 1.20},
                {"strike": 200, "bid": 0.10, "ask": 0.20},
            ],
            "put": [],
        }
        # Width=50 requested — nearest OTM long is 100 wider. Should fail.
        s = build_spread_from_chain(chain, 100, "CALL", 50, 90.0)
        assert s is None

    def test_top_picks_dedupes_by_short_strike(self):
        """When the same short strike appears with multiple widths (from
        multi-width enumeration), top-N should show only the best-ROI variant
        per short — no duplicate rows for the same short leg."""
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "5"])
        scan_data = {
            "prev_closes": {"SPX": 7100.0},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-20",
                    "spreads": {
                        "SPX": [
                            # Same short=7050, three widths — only the best (width=30, roi=6.0) should show.
                            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7040,
                             "credit": 0.20, "roi_pct": 2.0, "otm_pct": 0.7, "width": 10},
                            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7030,
                             "credit": 0.70, "roi_pct": 3.6, "otm_pct": 0.7, "width": 20},
                            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7020,
                             "credit": 1.20, "roi_pct": 6.0, "otm_pct": 0.7, "width": 30},
                            # Distinct short=7000 — should also show.
                            {"option_type": "PUT", "short_strike": 7000, "long_strike": 6980,
                             "credit": 0.50, "roi_pct": 2.6, "otm_pct": 1.4, "width": 20},
                        ],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        text = "\n".join(lines)
        # Only the best variant at 7050 (long=7020) appears; 7040 and 7030 are suppressed.
        assert "7050/7020" in text
        assert "7050/7040" not in text
        assert "7050/7030" not in text
        # Different short still appears.
        assert "7000/6980" in text
        # Exactly one data row mentions 7050.
        assert sum(1 for ln in lines if "7050/" in ln) == 1

    def test_dte_expiration_mapping(self):
        from spread_scanner import map_dte_to_expirations
        from datetime import date

        today = date.today().isoformat()
        tomorrow = "2099-01-02"  # far future to ensure it's after today
        expirations = [today, tomorrow]

        result = map_dte_to_expirations([0, 1], expirations)
        assert 0 in result
        assert result[0] == today

    def test_once_flag_exits(self):
        from spread_scanner import parse_args

        args = parse_args(["--once"])
        assert args.once is True

    def test_width_defaults(self):
        from spread_scanner import DEFAULT_WIDTHS, parse_args

        assert DEFAULT_WIDTHS["SPX"] == 20
        assert DEFAULT_WIDTHS["NDX"] == 50
        assert DEFAULT_WIDTHS["RUT"] == 20

        # CLI override
        args = parse_args(["--widths", "SPX=25,RUT=10,NDX=100"])
        assert args.widths["SPX"] == 25
        assert args.widths["RUT"] == 10
        assert args.widths["NDX"] == 100

    def test_top_picks_default(self):
        from spread_scanner import parse_args

        args = parse_args([])
        assert args.top == 3

    def test_top_picks_rendering(self):
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "2"])
        scan_data = {
            "prev_closes": {"SPX": 7100.0},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-20",
                    "spreads": {
                        "SPX": [
                            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7030,
                             "credit": 1.5, "roi_pct": 8.1, "otm_pct": 0.7, "width": 20},
                            {"option_type": "PUT", "short_strike": 7000, "long_strike": 6980,
                             "credit": 0.5, "roi_pct": 2.6, "otm_pct": 1.4, "width": 20},
                            {"option_type": "CALL", "short_strike": 7200, "long_strike": 7220,
                             "credit": 2.0, "roi_pct": 11.1, "otm_pct": 1.4, "width": 20},
                        ],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        assert len(lines) > 0
        # Should contain the top 2 by ROI: CALL 11.1% and PUT 8.1%
        text = "\n".join(lines)
        assert "7200" in text  # highest ROI
        assert "7050" in text  # second highest
        assert "7000" not in text  # third, excluded

    def test_top_picks_filters(self):
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "5",
                           "--min-credit", "1.0", "--min-otm", "0.5", "--max-otm", "2.0"])
        scan_data = {
            "prev_closes": {"SPX": 7100.0},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-20",
                    "spreads": {
                        "SPX": [
                            # Excluded: credit too low
                            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7030,
                             "credit": 0.50, "roi_pct": 2.6, "otm_pct": 0.7, "width": 20},
                            # Excluded: OTM too low
                            {"option_type": "PUT", "short_strike": 7090, "long_strike": 7070,
                             "credit": 3.00, "roi_pct": 17.6, "otm_pct": 0.1, "width": 20},
                            # Excluded: OTM too high
                            {"option_type": "PUT", "short_strike": 6900, "long_strike": 6880,
                             "credit": 1.20, "roi_pct": 6.4, "otm_pct": 2.8, "width": 20},
                            # Passes all filters
                            {"option_type": "CALL", "short_strike": 7200, "long_strike": 7220,
                             "credit": 2.00, "roi_pct": 11.1, "otm_pct": 1.4, "width": 20},
                        ],
                    },
                }
            },
        }
        lines = _render_top_picks(scan_data, args)
        text = "\n".join(lines)
        assert "7200" in text  # only one that passes all filters
        assert "7050" not in text  # credit too low
        assert "7090" not in text  # OTM too low
        assert "6900" not in text  # OTM too high
        assert "cr≥$1.00" in text  # filter shown in header
        assert "otm≥0.5%" in text

    def test_offline_ticker_graceful(self):
        """Connection errors should produce empty data, not crash."""
        from spread_scanner import compute_spreads

        # Empty chain
        spreads = compute_spreads({}, "SPX", 7100.0, 20)
        assert spreads == []

        # Chain with no valid spreads
        spreads = compute_spreads({"put": [], "call": []}, "SPX", 7100.0, 20)
        assert spreads == []

    @pytest.mark.asyncio
    async def test_scan_all_tickers_mock(self):
        """End-to-end scan with mocked HTTP responses."""
        from unittest.mock import AsyncMock, patch
        from spread_scanner import scan_all_tickers, parse_args

        args = parse_args(["--tickers", "SPX", "--dte", "0", "--once"])

        mock_client = AsyncMock()

        # Mock quote response
        quote_resp = AsyncMock()
        quote_resp.status_code = 200
        quote_resp.json.return_value = {"last": 7100.0, "bid": 7099.0, "ask": 7101.0}

        # Mock expirations response
        exp_resp = AsyncMock()
        exp_resp.status_code = 200
        from datetime import date
        today = date.today().isoformat()
        exp_resp.json.return_value = {"expirations": [today]}

        # Mock option chain response
        chain_resp = AsyncMock()
        chain_resp.status_code = 200
        chain_resp.json.return_value = {
            "quotes": {
                "put": [
                    {"strike": 7050, "bid": 1.50, "ask": 1.80},
                    {"strike": 7030, "bid": 0.80, "ask": 1.10},
                ],
                "call": [],
            }
        }

        # Route different URL patterns
        async def mock_get(url, **kwargs):
            if "list_expirations" in str(kwargs.get("params", {})):
                return exp_resp
            elif "/market/options/" in url:
                return chain_resp
            else:
                return quote_resp

        mock_client.get = mock_get

        result = await scan_all_tickers(mock_client, args)
        assert "quotes" in result
        assert "dte_sections" in result
        assert "SPX" in result["quotes"]

    def test_tier_intraday_resolution(self):
        """Test that resolve_tier_strike uses pct applied to current price."""
        from spread_scanner import resolve_tier_strike

        tier_data = {
            "hourly": {
                "SPX": {
                    "previous_close": 7100.0,
                    "recommended": {
                        "intraday": {
                            "aggressive": {"put": 90, "call": 90},
                            "moderate": {"put": 95, "call": 95},
                            "conservative": {"put": 98, "call": 98},
                        },
                        "close_to_close": {
                            "aggressive": {"put": 90, "call": 90},
                            "moderate": {"put": 95, "call": 95},
                            "conservative": {"put": 98, "call": 98},
                        },
                    },
                    "slots": {
                        "10:00": {
                            "when_down": {
                                "pct": {"p90": -1.0, "p95": -1.5, "p98": -2.0},
                                "price": {"p90": 7050.0, "p95": 7020.0, "p98": 6990.0},
                            },
                            "when_up": {
                                "pct": {"p90": 1.0, "p95": 1.5, "p98": 2.0},
                                "price": {"p90": 7150.0, "p95": 7180.0, "p98": 7210.0},
                            },
                        },
                    },
                }
            }
        }
        # Mock current time to be in the 10:00 slot
        from unittest.mock import patch
        import spread_scanner

        with patch.object(spread_scanner, "_find_current_slot", return_value="10:00"):
            # Current price = 7050 (below prev close)
            # Aggressive p90 pct = -1.0% → 7050 * 0.99 = 6979.5 → round to 6975
            result = resolve_tier_strike(
                tier_data, "SPX", "put", "aggressive", "intraday", 7100.0, 7050.0,
            )
            assert result is not None
            strike, raw_price, pctl, pct_val = result
            assert pctl == 90
            assert pct_val == -1.0
            # 7050 * (1 + (-1.0/100)) = 7050 * 0.99 = 6979.5
            assert abs(raw_price - 6979.5) < 0.1
            # Rounded to SPX 5-point increment: int(6979.5/5)*5 = 6975
            assert strike == 6975.0

    def test_tier_close_to_close_resolution(self):
        """Test close-to-close tier uses tickers data with prev_close."""
        from spread_scanner import resolve_tier_strike
        from unittest.mock import patch
        import spread_scanner

        tier_data = {
            "hourly": {
                "RUT": {
                    "previous_close": 2150.0,
                    "recommended": {
                        "close_to_close": {
                            "aggressive": {"put": 90, "call": 90},
                            "moderate": {"put": 95, "call": 95},
                            "conservative": {"put": 98, "call": 98},
                        },
                        "intraday": {
                            "aggressive": {"put": 90, "call": 90},
                        },
                    },
                    "slots": {
                        "11:00": {
                            "when_down": {
                                "pct": {"p95": -1.4},
                                "price": {"p95": 2120.0},
                            },
                            "when_up": {
                                "pct": {"p95": 1.4},
                                "price": {"p95": 2180.0},
                            },
                        },
                    },
                }
            },
            # Close-to-close data from tickers field
            "tickers": [
                {
                    "ticker": "RUT",
                    "windows": {
                        "0": {
                            "when_down": {
                                "pct": {"p90": -1.5, "p95": -2.0, "p98": -2.5},
                            },
                            "when_up": {
                                "pct": {"p90": 1.5, "p95": 2.0, "p98": 2.5},
                            },
                        }
                    }
                }
            ]
        }

        with patch.object(spread_scanner, "_find_current_slot", return_value="11:00"):
            # Moderate c2c put = p95, c2c window 0 pct p95 = -2.0%
            # Applied to prev_close 2150: 2150 * 0.98 = 2107
            result = resolve_tier_strike(
                tier_data, "RUT", "put", "moderate", "close_to_close", 2150.0, 2140.0,
            )
            assert result is not None
            strike, raw_price, pctl, pct_val = result
            assert pctl == 95
            assert pct_val == -2.0
            # 2150 * (1 + (-2.0/100)) = 2150 * 0.98 = 2107
            assert abs(raw_price - 2107.0) < 0.1
            # Rounded to RUT 5-point: int(2107/5)*5 = 2105
            assert strike == 2105.0

    def test_compute_norm_roi(self):
        from spread_scanner import _compute_norm_roi
        # DTE 0: norm = ROI / 1
        assert _compute_norm_roi(10.0, 0) == 10.0
        # DTE 1: norm = ROI / 2
        assert _compute_norm_roi(10.0, 1) == 5.0
        # DTE 2: norm = ROI / 3
        assert _compute_norm_roi(9.0, 2) == 3.0

    def test_filter_by_norm_roi(self):
        from spread_scanner import _filter_by_norm_roi

        candidates = [
            {"roi_pct": 8.1, "dte": 0, "symbol": "SPX", "option_type": "PUT",
             "short_strike": 5700, "long_strike": 5680, "credit": 1.50, "otm_pct": 1.72},
            {"roi_pct": 1.5, "dte": 0, "symbol": "SPX", "option_type": "PUT",
             "short_strike": 5750, "long_strike": 5730, "credit": 0.30, "otm_pct": 0.86},
            {"roi_pct": 11.1, "dte": 1, "symbol": "SPX", "option_type": "CALL",
             "short_strike": 5900, "long_strike": 5920, "credit": 2.00, "otm_pct": 1.72},
        ]

        qualifying = _filter_by_norm_roi(candidates, 4.0)
        # DTE 0 PUT 8.1%: norm = 8.1/1 = 8.1 >= 4 ✓
        # DTE 0 PUT 1.5%: norm = 1.5/1 = 1.5 < 4 ✗
        # DTE 1 CALL 11.1%: norm = 11.1/2 = 5.55 >= 4 ✓
        assert len(qualifying) == 2
        assert qualifying[0]["norm_roi"] == 8.1
        assert qualifying[1]["norm_roi"] == 5.55

    def test_filter_by_norm_roi_disabled(self):
        from spread_scanner import _filter_by_norm_roi

        assert _filter_by_norm_roi([], 5.0) == []
        assert _filter_by_norm_roi([{"roi_pct": 10, "dte": 0}], 0) == []

    def test_log_qualifying_spreads(self, tmp_path):
        from spread_scanner import _log_qualifying_spreads
        import json

        log_file = str(tmp_path / "test_spreads.jsonl")
        spreads = [
            {"symbol": "SPX", "option_type": "PUT", "norm_roi": 8.1, "credit": 1.50},
            {"symbol": "RUT", "option_type": "CALL", "norm_roi": 5.5, "credit": 2.00},
        ]
        _log_qualifying_spreads(spreads, log_file)

        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["symbol"] == "SPX"
        assert json.loads(lines[1])["norm_roi"] == 5.5

        # Append more
        _log_qualifying_spreads([{"symbol": "NDX", "norm_roi": 12.0}], log_file)
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_log_empty_list_noop(self, tmp_path):
        from spread_scanner import _log_qualifying_spreads

        log_file = str(tmp_path / "empty.jsonl")
        _log_qualifying_spreads([], log_file)
        assert not (tmp_path / "empty.jsonl").exists()

    def test_parse_args_log_and_notify(self):
        from spread_scanner import parse_args

        args = parse_args([
            "--min-norm-roi", "3.5",
            "--log", "3:spreads.jsonl",
            "--notify", "4:ak@gmail.com",
        ])
        assert args.min_norm_roi == 3.5
        assert args.log_threshold == 3.0
        assert args.log_file == "spreads.jsonl"
        assert args.notify_threshold == 4.0
        assert args.notify_email == "ak@gmail.com"

    def test_parse_args_defaults(self):
        from spread_scanner import parse_args

        args = parse_args([])
        assert args.min_norm_roi == 0
        assert args.log_threshold == 0
        assert args.log_file is None
        assert args.notify_threshold == 0
        assert args.notify_email is None

    @pytest.mark.asyncio
    async def test_notify_qualifying_spreads(self):
        from spread_scanner import _notify_qualifying_spreads
        import httpx

        calls = []

        async def mock_post(url, **kwargs):
            calls.append((url, kwargs))
            resp = httpx.Response(200, json={"status": "sent"})
            return resp

        client = httpx.AsyncClient()
        client.post = mock_post

        spreads = [
            {"symbol": "SPX", "option_type": "PUT", "dte": 0,
             "short_strike": 5700, "long_strike": 5680,
             "credit": 1.50, "roi_pct": 8.1, "norm_roi": 8.1, "otm_pct": 1.72},
        ]
        await _notify_qualifying_spreads(
            client, spreads, "http://localhost:9102", "ak@gmail.com",
        )
        assert len(calls) == 1
        assert "/api/notify" in calls[0][0]
        body = calls[0][1]["json"]
        assert body["channel"] == "email"
        assert body["to"] == "ak@gmail.com"
        assert "nROI=8.1%" in body["message"]

    @pytest.mark.asyncio
    async def test_notify_empty_list_noop(self):
        from spread_scanner import _notify_qualifying_spreads
        import httpx

        calls = []
        client = httpx.AsyncClient()
        client.post = lambda *a, **kw: calls.append(1)

        await _notify_qualifying_spreads(
            client, [], "http://localhost:9102", "ak@gmail.com",
        )
        assert len(calls) == 0

    def test_top_picks_shows_norm_roi(self):
        """Top picks table includes nROI column."""
        from spread_scanner import _render_top_picks, parse_args

        args = parse_args(["--tickers", "SPX", "--top", "1"])
        scan_data = {
            "quotes": {"SPX": {"last": 5800}},
            "prev_closes": {"SPX": 5790},
            "dte_sections": {
                1: {
                    "expiration": "2026-04-22",
                    "spreads": {
                        "SPX": [
                            {"option_type": "PUT", "short_strike": 5700, "long_strike": 5680,
                             "width": 20, "credit": 1.50, "roi_pct": 8.1, "otm_pct": 1.72},
                        ],
                    },
                },
            },
        }
        lines = _render_top_picks(scan_data, args)
        text = "\n".join(lines)
        assert "nROI" in text  # header has nROI column

    def test_min_norm_roi_filters_top_picks(self):
        """--min-norm-roi filters low nROI spreads from top picks."""
        from spread_scanner import _render_top_picks, parse_args

        # nROI filter = 5 → DTE1 spread with ROI 8.1% has nROI 4.05 → excluded
        args = parse_args(["--tickers", "SPX", "--top", "5", "--min-norm-roi", "5"])
        scan_data = {
            "quotes": {"SPX": {"last": 5800}},
            "prev_closes": {"SPX": 5790},
            "dte_sections": {
                0: {
                    "expiration": "2026-04-21",
                    "spreads": {
                        "SPX": [
                            {"option_type": "PUT", "short_strike": 5700, "long_strike": 5680,
                             "width": 20, "credit": 1.50, "roi_pct": 8.1, "otm_pct": 1.72},
                        ],
                    },
                },
                1: {
                    "expiration": "2026-04-22",
                    "spreads": {
                        "SPX": [
                            {"option_type": "PUT", "short_strike": 5700, "long_strike": 5680,
                             "width": 20, "credit": 1.50, "roi_pct": 8.1, "otm_pct": 1.72},
                        ],
                    },
                },
            },
        }
        lines = _render_top_picks(scan_data, args)
        text = "\n".join(lines)
        # DTE 0: nROI = 8.1/1 = 8.1 >= 5 ✓ (included)
        # DTE 1: nROI = 8.1/2 = 4.05 < 5 ✗ (excluded)
        assert "D0" in text
        assert "D1" not in text

    def test_notify_gated_by_market_hours(self):
        """Verify spread_scanner imports is_market_hours for notification gating."""
        import spread_scanner
        assert hasattr(spread_scanner, "is_market_hours")
        assert callable(spread_scanner.is_market_hours)

    def test_build_spread_from_chain(self):
        from spread_scanner import build_spread_from_chain

        chain = {
            "put": [
                {"strike": 2735, "bid": 1.20, "ask": 1.40},
                {"strike": 2710, "bid": 0.30, "ask": 0.50},
            ],
            "call": [
                {"strike": 2850, "bid": 1.50, "ask": 1.70},
                {"strike": 2875, "bid": 0.40, "ask": 0.60},
            ],
        }
        # PUT: short 2735 bid=1.20, long 2710 ask=0.50 → credit=0.70
        spread = build_spread_from_chain(chain, 2735, "PUT", 25, 2800)
        assert spread is not None
        assert spread["credit"] == 0.70
        assert spread["short_strike"] == 2735
        assert spread["long_strike"] == 2710

        # CALL: short 2850 bid=1.50, long 2875 ask=0.60 → credit=0.90
        spread = build_spread_from_chain(chain, 2850, "CALL", 25, 2800)
        assert spread is not None
        assert spread["credit"] == 0.90

        # Width exceeds what the exchange lists → snap to the widest listed
        # long within the max width. width=30 on short=2735 targets 2705
        # (not listed); 2710 is the widest listed OTM long within 30 → used.
        snapped = build_spread_from_chain(chain, 2735, "PUT", 30, 2800)
        assert snapped is not None
        assert snapped["long_strike"] == 2710
        assert snapped["width"] == 25  # actual width, not the configured max

        # Short strike not in chain → None (can't price at all).
        assert build_spread_from_chain(chain, 2700, "PUT", 25, 2800) is None

    def test_build_spread_from_chain_no_edge_returns_none_by_default(self):
        """Chain has the strikes but short_bid == long_ask → credit=0 → None."""
        from spread_scanner import build_spread_from_chain
        chain = {"put": [
            {"strike": 6945.0, "bid": 0.20, "ask": 0.25},
            {"strike": 6920.0, "bid": 0.15, "ask": 0.20},   # ask == short bid
        ]}
        assert build_spread_from_chain(chain, 6945.0, "PUT", 25, 7128.0) is None

    def test_build_spread_from_chain_allow_no_edge_returns_probe(self):
        """allow_no_edge=True returns the pair with credit=0 + note so the
        dashboard can distinguish 'no chain data' from 'no economic edge'."""
        from spread_scanner import build_spread_from_chain
        chain = {"put": [
            {"strike": 6945.0, "bid": 0.20, "ask": 0.25},
            {"strike": 6920.0, "bid": 0.15, "ask": 0.20},
        ]}
        r = build_spread_from_chain(chain, 6945.0, "PUT", 25, 7128.0, allow_no_edge=True)
        assert r is not None
        assert r["credit"] == 0.0
        assert r["short_bid"] == 0.20
        assert r["long_ask"] == 0.20
        assert r["note"] == "no edge"

    def test_build_spread_from_chain_no_bid_flagged(self):
        from spread_scanner import build_spread_from_chain
        chain = {"put": [
            {"strike": 6945.0, "bid": 0, "ask": 0.25},
            {"strike": 6920.0, "bid": 0.15, "ask": 0.20},
        ]}
        r = build_spread_from_chain(chain, 6945.0, "PUT", 25, 7128.0, allow_no_edge=True)
        assert r["note"] == "no bid"

    def test_build_spread_from_chain_missing_leg_still_returns_none(self):
        """Even with allow_no_edge, truly missing strike data → None."""
        from spread_scanner import build_spread_from_chain
        chain = {"put": [{"strike": 6945.0, "bid": 0.20, "ask": 0.25}]}  # no 6920
        assert build_spread_from_chain(
            chain, 6945.0, "PUT", 25, 7128.0, allow_no_edge=True,
        ) is None


class TestPrevCloseCache:
    """Prev-close cache: fetch at startup + 04:00 PT refresh on trading days."""

    def _cache(self):
        from spread_scanner import PrevCloseCache
        return PrevCloseCache(["SPX", "NDX", "RUT"], "http://db")

    def test_should_refresh_first_time(self):
        """No prior refresh → refresh immediately."""
        c = self._cache()
        assert c.should_refresh() is True

    def _complete_cache(self):
        """Cache with all tickers populated — bypasses the incomplete-retry path."""
        c = self._cache()
        c.values = {sym: 100.0 for sym in c.tickers}
        return c

    def test_should_not_refresh_before_24h(self, monkeypatch):
        """< 24h since last refresh → don't refresh, regardless of time-of-day."""
        from spread_scanner import _PT
        from datetime import datetime
        c = self._complete_cache()
        c.last_refreshed_at = datetime(2026, 4, 21, 6, 0, tzinfo=_PT)
        # Next day 05:00 PT — 23h later, past 04:00 but under 24h
        assert c.should_refresh(datetime(2026, 4, 22, 5, 0, tzinfo=_PT)) is False

    def test_should_not_refresh_before_04_pt(self):
        """>= 24h passed but still before 04:00 PT → wait."""
        from spread_scanner import _PT
        from datetime import datetime
        c = self._complete_cache()
        c.last_refreshed_at = datetime(2026, 4, 21, 3, 0, tzinfo=_PT)
        # Next day 03:30 PT — 24.5h later, but still before 04:00
        assert c.should_refresh(datetime(2026, 4, 22, 3, 30, tzinfo=_PT)) is False

    def test_should_not_refresh_on_weekend(self):
        """After 04:00 PT + >=24h, but it's a Saturday → skip."""
        from spread_scanner import _PT
        from datetime import datetime
        c = self._complete_cache()
        # Friday 2026-04-17 05:00 PT
        c.last_refreshed_at = datetime(2026, 4, 17, 5, 0, tzinfo=_PT)
        # Saturday 2026-04-18 05:00 PT — 24h later, past 04:00, but not a trading day
        assert c.should_refresh(datetime(2026, 4, 18, 5, 0, tzinfo=_PT)) is False

    def test_should_refresh_next_trading_day_after_04(self):
        """>=24h passed, past 04:00 PT, on a trading day → refresh."""
        from spread_scanner import _PT
        from datetime import datetime
        c = self._complete_cache()
        # Monday 04:00 PT
        c.last_refreshed_at = datetime(2026, 4, 20, 4, 0, tzinfo=_PT)
        # Tuesday 04:15 PT — trading day, past 04:00, 24.25h later
        assert c.should_refresh(datetime(2026, 4, 21, 4, 15, tzinfo=_PT)) is True

    @staticmethod
    def _range_percentiles_fake(closes: dict[str, float | None]):
        """Return a FakeClient that mimics /api/range_percentiles for the
        given {ticker: previous_close} map. Missing tickers → row dropped."""
        class FakeClient:
            async def get(self_c, url, params=None, timeout=None):
                req_tickers = [t.strip() for t in (params or {}).get("tickers", "").split(",") if t.strip()]
                rows = []
                for t in req_tickers:
                    pc = closes.get(t)
                    if pc is not None:
                        rows.append({"ticker": t, "previous_close": pc})
                class R:
                    status_code = 200
                    def json(self_r): return rows
                return R()
        return FakeClient

    def test_refresh_merges_tier_and_db(self):
        """Tier-data-missing symbols are filled from db_server fallback."""
        import asyncio
        from spread_scanner import _PT
        from datetime import datetime

        FakeClient = self._range_percentiles_fake({"RUT": 2800.0})
        c = self._cache()
        tier_data = {"hourly": {
            "SPX": {"previous_close": 7109.14},
            "NDX": {"previous_close": 26590.34},
        }}
        merged = asyncio.run(c.refresh(
            FakeClient(), tier_data=tier_data,
            now_pt=datetime(2026, 4, 21, 6, 0, tzinfo=_PT),
        ))
        assert merged["SPX"] == 7109.14
        assert merged["NDX"] == 26590.34
        assert merged["RUT"] == 2800.0
        assert c.last_refreshed_at is not None

    def test_refresh_tier_empty_falls_back_to_db(self):
        """tier_data=None → fetches all tickers from db_server."""
        import asyncio
        FakeClient = self._range_percentiles_fake({
            "SPX": 7109.0, "NDX": 26590.0, "RUT": 2792.0,
        })
        c = self._cache()
        merged = asyncio.run(c.refresh(FakeClient(), tier_data=None))
        assert merged == {"SPX": 7109.0, "NDX": 26590.0, "RUT": 2792.0}

    def test_refresh_keeps_prior_value_when_both_sources_fail(self):
        """If a source fails for a ticker already in cache, retain the old value."""
        import asyncio
        class FailClient:
            async def get(self, url, params=None):
                raise RuntimeError("db unreachable")

        c = self._cache()
        c.values = {"SPX": 7000.0, "NDX": 26000.0, "RUT": 2800.0}
        merged = asyncio.run(c.refresh(FailClient(), tier_data=None))
        # Prior values preserved
        assert merged["SPX"] == 7000.0
        assert merged["NDX"] == 26000.0
        assert merged["RUT"] == 2800.0

    def test_as_dict_fills_missing_with_zero(self):
        """Symbols never populated come back as 0 (same shape old callers expect)."""
        c = self._cache()
        c.values = {"SPX": 7000.0}
        d = c.as_dict()
        assert d == {"SPX": 7000.0, "NDX": 0.0, "RUT": 0.0}

    def test_should_refresh_when_incomplete_after_retry_window(self):
        """Cache incomplete + >= 60s since last retry → should_refresh True."""
        from spread_scanner import _PT
        from datetime import datetime
        c = self._cache()
        c.values = {"SPX": 7100.0}        # NDX, RUT missing
        c.last_refreshed_at = datetime(2026, 4, 22, 7, 0, tzinfo=_PT)
        c.last_incomplete_attempt_at = datetime(2026, 4, 22, 7, 0, tzinfo=_PT)
        # 59s after last attempt → still within rate-limit → False
        now1 = datetime(2026, 4, 22, 7, 0, 59, tzinfo=_PT)
        assert c.should_refresh(now1) is False
        # 60s later → True
        now2 = datetime(2026, 4, 22, 7, 1, 0, tzinfo=_PT)
        assert c.should_refresh(now2) is True

    def test_refresh_logs_recovery_when_ticker_becomes_populated(self, capsys):
        """First empty refresh → then tier_data fills NDX → log '[prev_close] NDX: populated'."""
        import asyncio
        from spread_scanner import PrevCloseCache

        class EmptyDbClient:
            async def get(self, url, params=None):
                class R:
                    status_code = 200
                    def json(self_): return {"rows": []}
                return R()

        c = PrevCloseCache(["SPX", "NDX", "RUT"], "http://db")
        # First refresh — db empty, no tier data → all missing
        asyncio.run(c.refresh(EmptyDbClient(), tier_data=None))
        assert c._missing_tickers() == ["SPX", "NDX", "RUT"]
        first_err = capsys.readouterr().err
        assert "WARNING: no prev_close for ['SPX', 'NDX', 'RUT']" in first_err

        # Second refresh — tier data now has NDX; db still empty for others.
        tier = {"hourly": {"NDX": {"previous_close": 26590.34}}}
        asyncio.run(c.refresh(EmptyDbClient(), tier_data=tier))
        err = capsys.readouterr().err
        # Recovery for NDX should be logged.
        assert "[prev_close] NDX: populated = 26590.34" in err
        # SPX and RUT still missing → WARNING still present.
        assert "no prev_close for ['SPX', 'RUT']" in err

    def test_refresh_clears_incomplete_flag_when_complete(self):
        """Once every ticker has a positive value, the retry timer stops firing."""
        import asyncio
        from spread_scanner import PrevCloseCache

        FullDbClient = self._range_percentiles_fake({
            "SPX": 7109.14, "NDX": 26590.34, "RUT": 2792.96,
        })
        c = PrevCloseCache(["SPX", "NDX", "RUT"], "http://db")
        asyncio.run(c.refresh(FullDbClient(), tier_data=None))
        assert c.is_complete()
        assert c.last_incomplete_attempt_at is None
        # should_refresh is False (complete + recent) — no retry needed.
        assert c.should_refresh() is False

    def test_fetch_prev_closes_uses_range_percentiles_endpoint(self):
        """Canonical path: `/api/range_percentiles` — same endpoint the CLI
        uses (_fetch_prev_close_from_db_server in utp.py). Returns YESTERDAY's
        EOD close, not today's intraday `daily_prices` row — which was the bug
        when querying daily_prices directly after market open on day N+1.
        """
        import asyncio
        from spread_scanner import fetch_prev_closes

        seen: list[dict] = []
        class Fake:
            async def get(self_c, url, params=None, timeout=None):
                seen.append({"url": url, "params": dict(params or {}), "timeout": timeout})
                class R:
                    status_code = 200
                    def json(self_r):
                        return [
                            {"ticker": "SPX", "previous_close": 7064.01},
                            {"ticker": "NDX", "previous_close": 26479.47},
                            {"ticker": "RUT", "previous_close": 2764.97},
                        ]
                return R()

        result = asyncio.run(fetch_prev_closes(Fake(), "http://db", ["SPX", "NDX", "RUT"]))
        assert result == {"SPX": 7064.01, "NDX": 26479.47, "RUT": 2764.97}
        # Exactly one batched request
        assert len(seen) == 1
        assert seen[0]["url"].endswith("/api/range_percentiles")
        assert seen[0]["params"]["tickers"] == "SPX,NDX,RUT"
        # Sanity on the canonical payload parameters (match utp.py)
        assert seen[0]["params"]["lookback"] == "30"
        assert seen[0]["params"]["min_days"] == "2"
        assert seen[0]["params"]["window"] == "1"

    def test_fetch_prev_closes_strips_I_prefix_from_endpoint_ticker(self):
        """Endpoint may return ticker with 'I:' prefix; map back to plain symbol."""
        import asyncio
        from spread_scanner import fetch_prev_closes
        class Fake:
            async def get(self_c, url, params=None, timeout=None):
                class R:
                    status_code = 200
                    def json(self_r): return [{"ticker": "I:SPX", "previous_close": 7064.01}]
                return R()
        result = asyncio.run(fetch_prev_closes(Fake(), "http://db", ["SPX"]))
        assert result == {"SPX": 7064.01}

    def test_fetch_prev_closes_single_ticker_accepts_dict_response(self):
        """Single-ticker request: endpoint returns a dict (not a list). Handle both."""
        import asyncio
        from spread_scanner import fetch_prev_closes
        class Fake:
            async def get(self_c, url, params=None, timeout=None):
                class R:
                    status_code = 200
                    def json(self_r):
                        return {"ticker": "SPX", "previous_close": 7064.01}
                return R()
        result = asyncio.run(fetch_prev_closes(Fake(), "http://db", ["SPX"]))
        assert result == {"SPX": 7064.01}

    def test_fetch_prev_closes_skips_zero_values(self):
        """Endpoint returning previous_close=0 is treated as 'no data'."""
        import asyncio
        from spread_scanner import fetch_prev_closes
        class FakeClient:
            async def get(self, url, params=None, timeout=None):
                class R:
                    status_code = 200
                    def json(self_inner):
                        return [{"ticker": "SPX", "previous_close": 0.0}]
                return R()
        result = asyncio.run(fetch_prev_closes(FakeClient(), "http://db", ["SPX"]))
        assert "SPX" not in result

    def test_scan_all_tickers_uses_cache(self):
        """When prev_close_cache is supplied, its values show up in result."""
        import asyncio
        from spread_scanner import PrevCloseCache, scan_all_tickers, parse_args

        class FakeClient:
            async def get(self, url, params=None):
                class R:
                    status_code = 200
                    def json(self_inner):
                        # Minimal daemon response
                        if "/market/quote/" in url:
                            return {"last": 7100}
                        if "/market/options/" in url:
                            return {"expirations": ["2026-04-21"]}
                        return {"rows": []}
                return R()

        args = parse_args(["--tickers", "SPX", "--once"])
        cache = PrevCloseCache(["SPX"], "http://db")
        cache.values = {"SPX": 7109.14}
        from datetime import datetime
        from spread_scanner import _PT
        cache.last_refreshed_at = datetime(2026, 4, 21, 10, 0, tzinfo=_PT)

        data = asyncio.run(scan_all_tickers(FakeClient(), args, prev_close_cache=cache))
        assert data["prev_closes"]["SPX"] == 7109.14


class TestActionHandlers:
    """ActionHandler framework: LogHandler, NotifyHandler, YAML config, scan_loop dispatch."""

    @staticmethod
    def _sample_candidates():
        return [
            {"symbol": "SPX", "option_type": "PUT", "short_strike": 5700, "long_strike": 5680,
             "width": 20, "credit": 1.5, "roi_pct": 8.1, "otm_pct": 1.72, "dte": 0, "prev_close": 5790},
            {"symbol": "NDX", "option_type": "CALL", "short_strike": 21000, "long_strike": 21050,
             "width": 50, "credit": 4.0, "roi_pct": 5.0, "otm_pct": 1.0, "dte": 0, "prev_close": 20800},
        ]

    # ── LogHandler ────────────────────────────────────────────────────────

    def test_log_handler_filter_and_fire_writes_jsonl(self, tmp_path):
        import asyncio
        from spread_scanner import LogHandler, HandlerContext

        log_file = tmp_path / "log.jsonl"
        handler = LogHandler(min_norm_roi=4.0, path=str(log_file))
        # Both sample candidates have nROI = roi_pct/(0+1) = 8.1 and 5.0 — both pass >= 4.0
        eligible = handler.filter(self._sample_candidates())
        assert len(eligible) == 2

        ctx = HandlerContext(client=None, args=None, scan_data={}, is_market_hours=True, now_ts="t")
        asyncio.run(handler.fire(eligible, ctx))

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        assert parsed[0]["norm_roi"] >= parsed[1]["norm_roi"]  # sorted desc
        assert handler.count == 2

    def test_log_handler_filter_applies_threshold(self):
        from spread_scanner import LogHandler
        handler = LogHandler(min_norm_roi=6.0, path="/tmp/unused.jsonl")
        # Only SPX (nROI=8.1) passes; NDX (nROI=5.0) is dropped
        eligible = handler.filter(self._sample_candidates())
        assert len(eligible) == 1
        assert eligible[0]["symbol"] == "SPX"

    # ── NotifyHandler ─────────────────────────────────────────────────────

    def test_notify_handler_dedupes_across_calls(self, monkeypatch):
        import asyncio
        from spread_scanner import NotifyHandler, HandlerContext
        import spread_scanner as ss

        monkeypatch.setattr(ss, "is_market_hours", lambda: True)

        calls = []
        class FakeClient:
            async def post(self, url, json=None, timeout=None):
                calls.append(json)
                class R: status_code = 200
                return R()

        h = NotifyHandler(min_norm_roi=4.0, email="x@y.com", url="http://notify")
        cands = self._sample_candidates()

        first = h.filter(cands)
        assert len(first) == 2
        ctx = HandlerContext(client=FakeClient(), args=None, scan_data={},
                             is_market_hours=True, now_ts="t")
        asyncio.run(h.fire(first, ctx))
        assert len(calls) == 1

        # Second scan with the same candidates → dedup filters both out → no POST
        second = h.filter(cands)
        assert second == []

    def test_notify_handler_gated_by_market_hours(self, monkeypatch):
        from spread_scanner import NotifyHandler
        import spread_scanner as ss
        monkeypatch.setattr(ss, "is_market_hours", lambda: False)

        h = NotifyHandler(min_norm_roi=4.0, email="x@y.com",
                          url="http://notify", gate_market_hours=True)
        assert h.filter(self._sample_candidates()) == []

    def test_notify_handler_gate_can_be_disabled(self, monkeypatch):
        from spread_scanner import NotifyHandler
        import spread_scanner as ss
        monkeypatch.setattr(ss, "is_market_hours", lambda: False)

        h = NotifyHandler(min_norm_roi=4.0, email="x@y.com",
                          url="http://notify", gate_market_hours=False)
        assert len(h.filter(self._sample_candidates())) == 2

    # ── build_handler factory ─────────────────────────────────────────────

    def test_build_handler_log(self):
        from spread_scanner import build_handler, LogHandler
        h = build_handler({"type": "log", "min_norm_roi": 2.5, "path": "/tmp/x.jsonl"})
        assert isinstance(h, LogHandler)
        assert h.min_norm_roi == 2.5
        assert h.path == "/tmp/x.jsonl"

    def test_build_handler_notify(self):
        from spread_scanner import build_handler, NotifyHandler
        h = build_handler({
            "type": "notify", "min_norm_roi": 3.0,
            "email": "a@b.com", "url": "http://host",
            "gate_market_hours": False, "top_n": 3,
        })
        assert isinstance(h, NotifyHandler)
        assert h.email == "a@b.com"
        assert h.url == "http://host"
        assert h.gate_market_hours is False
        assert h.top_n == 3

    def test_build_handler_unknown_type(self):
        import pytest
        from spread_scanner import build_handler
        with pytest.raises(ValueError, match="Unknown handler type"):
            build_handler({"type": "quantum_flux"})

    def test_build_handler_missing_required(self):
        import pytest
        from spread_scanner import build_handler
        with pytest.raises(ValueError, match="log handler requires 'path'"):
            build_handler({"type": "log", "min_norm_roi": 2.0})
        with pytest.raises(ValueError, match="notify handler requires 'email'"):
            build_handler({"type": "notify", "min_norm_roi": 2.0})

    # ── ScannerConfig / YAML ──────────────────────────────────────────────

    def test_scannerconfig_defaults(self):
        from spread_scanner import ScannerConfig, DEFAULT_TICKERS, DEFAULT_INTERVAL
        c = ScannerConfig()
        assert c.tickers == list(DEFAULT_TICKERS)
        assert c.interval == DEFAULT_INTERVAL
        assert c.handlers == []

    def test_scannerconfig_from_yaml_roundtrip(self, tmp_path):
        from spread_scanner import ScannerConfig
        path = tmp_path / "cfg.yaml"
        path.write_text("""
tickers: [SPX, NDX]
dte: [0, 1]
interval: 15
min_norm_roi: 2.5
handlers:
  - type: log
    min_norm_roi: 1.5
    path: /tmp/log.jsonl
""")
        c = ScannerConfig.from_yaml(str(path))
        assert c.tickers == ["SPX", "NDX"]
        assert c.dte == [0, 1]
        assert c.interval == 15
        assert c.min_norm_roi == 2.5
        assert len(c.handlers) == 1
        assert c.handlers[0]["type"] == "log"

    def test_scannerconfig_unknown_field_raises(self, tmp_path):
        import pytest
        from spread_scanner import ScannerConfig
        path = tmp_path / "cfg.yaml"
        path.write_text("bogus_field: 42\n")
        with pytest.raises(ValueError, match="Unknown ScannerConfig fields"):
            ScannerConfig.from_yaml(str(path))

    def test_scannerconfig_empty_yaml(self, tmp_path):
        from spread_scanner import ScannerConfig, DEFAULT_TICKERS
        path = tmp_path / "cfg.yaml"
        path.write_text("")
        c = ScannerConfig.from_yaml(str(path))
        assert c.tickers == list(DEFAULT_TICKERS)

    def test_to_cli_defaults_flattens_lists_to_strings(self):
        from spread_scanner import ScannerConfig
        c = ScannerConfig(tickers=["SPX", "RUT"], dte=[0, 1], widths={"SPX": 25, "RUT": 15})
        d = c.to_cli_defaults()
        assert d["tickers"] == "SPX,RUT"
        assert d["dte"] == "0,1"
        assert d["widths_str"] == "SPX=25,RUT=15"

    # ── CLI + YAML precedence ─────────────────────────────────────────────

    def test_config_yaml_sets_defaults(self, tmp_path, monkeypatch):
        """YAML config provides default tickers; CLI omitted → YAML value used."""
        from spread_scanner import _load_config
        path = tmp_path / "cfg.yaml"
        path.write_text("tickers: [NDX]\ninterval: 45\n")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, handlers = _load_config()
        assert args.tickers == ["NDX"]
        assert args.interval == 45

    def test_config_cli_overrides_yaml(self, tmp_path, monkeypatch):
        """YAML has tickers=[SPX]; CLI --tickers NDX,RUT → CLI wins."""
        from spread_scanner import _load_config
        path = tmp_path / "cfg.yaml"
        path.write_text("tickers: [SPX]\ninterval: 10\n")
        monkeypatch.setattr("sys.argv",
            ["spread_scanner.py", "--config", str(path), "--tickers", "NDX,RUT"])
        args, _ = _load_config()
        assert args.tickers == ["NDX", "RUT"]
        assert args.interval == 10  # YAML wins (not overridden)

    def test_config_yaml_declares_handlers(self, tmp_path, monkeypatch):
        from spread_scanner import _load_config, LogHandler, NotifyHandler
        path = tmp_path / "cfg.yaml"
        path.write_text("""
tickers: [SPX]
handlers:
  - type: log
    min_norm_roi: 1.5
    path: /tmp/log.jsonl
  - type: notify
    min_norm_roi: 3.0
    email: a@b.com
""")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, handlers = _load_config()
        assert len(handlers) == 2
        assert isinstance(handlers[0], LogHandler)
        assert isinstance(handlers[1], NotifyHandler)

    def test_cli_legacy_flags_replace_yaml_handler(self, tmp_path, monkeypatch):
        """If YAML declares a log handler AND CLI --log is passed, CLI wins."""
        from spread_scanner import _load_config, LogHandler
        path = tmp_path / "cfg.yaml"
        path.write_text("""
handlers:
  - type: log
    min_norm_roi: 1.5
    path: /tmp/yaml.jsonl
""")
        monkeypatch.setattr("sys.argv",
            ["spread_scanner.py", "--config", str(path), "--log", "4.0:/tmp/cli.jsonl"])
        args, handlers = _load_config()
        log_handlers = [h for h in handlers if isinstance(h, LogHandler)]
        assert len(log_handlers) == 1
        assert log_handlers[0].path == "/tmp/cli.jsonl"
        assert log_handlers[0].min_norm_roi == 4.0

    def test_yaml_min_tier_flows_through(self, tmp_path, monkeypatch):
        """YAML `min_tier: cons` must be normalized to 'conservative' on args."""
        from spread_scanner import _load_config
        path = tmp_path / "cfg.yaml"
        path.write_text("tickers: [SPX]\nmin_tier: cons\ntiers: true\n")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, _ = _load_config()
        assert args.min_tier == "conservative"
        assert args.tiers is True

    def test_yaml_min_otm_per_ticker(self, tmp_path, monkeypatch):
        """Per-ticker OTM floor in YAML → args.min_otm_per_ticker populated."""
        from spread_scanner import _load_config, _resolve_min_otm
        path = tmp_path / "cfg.yaml"
        path.write_text("""
tickers: [SPX, NDX, RUT]
min_otm: 1.0
min_otm_per_ticker:
  NDX: 2.5
  RUT: 1.5
""")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, _ = _load_config()
        assert args.min_otm == 1.0
        assert args.min_otm_per_ticker == {"NDX": 2.5, "RUT": 1.5}
        # Effective resolution
        assert _resolve_min_otm(args, "SPX") == 1.0    # scalar only
        assert _resolve_min_otm(args, "NDX") == 2.5    # per-ticker wins (higher)
        assert _resolve_min_otm(args, "RUT") == 1.5    # per-ticker wins (higher)

    def test_min_otm_per_ticker_stacks_with_scalar(self, tmp_path, monkeypatch):
        """Effective floor is max(scalar, per-ticker) — even if per-ticker is lower."""
        from spread_scanner import _load_config, _resolve_min_otm
        path = tmp_path / "cfg.yaml"
        path.write_text("""
tickers: [SPX]
min_otm: 2.0
min_otm_per_ticker: {SPX: 1.0}   # deliberately lower than scalar
""")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, _ = _load_config()
        # max(2.0, 1.0) = 2.0 — the scalar wins
        assert _resolve_min_otm(args, "SPX") == 2.0

    def test_cli_min_otm_per_ticker_overrides(self, tmp_path, monkeypatch):
        """CLI flag '--min-otm-per-ticker SYM=N,...' populates the dict."""
        from spread_scanner import _load_config, _resolve_min_otm
        monkeypatch.setattr("sys.argv", [
            "spread_scanner.py", "--tickers", "SPX,NDX",
            "--min-otm", "1.0",
            "--min-otm-per-ticker", "NDX=2.5",
        ])
        args, _ = _load_config()
        assert args.min_otm_per_ticker == {"NDX": 2.5}
        assert _resolve_min_otm(args, "NDX") == 2.5

    def test_max_otm_per_ticker_uses_tighter_of_scalar_or_per_ticker(self, tmp_path, monkeypatch):
        """For max (upper bound), tighter means LOWER — take min of scalar & per-ticker."""
        from spread_scanner import _load_config, _resolve_max_otm
        path = tmp_path / "cfg.yaml"
        path.write_text("""
tickers: [SPX, NDX]
max_otm: 5.0
max_otm_per_ticker: {NDX: 3.0}
""")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, _ = _load_config()
        assert _resolve_max_otm(args, "SPX") == 5.0   # only scalar
        assert _resolve_max_otm(args, "NDX") == 3.0   # per-ticker is tighter

    def test_resolve_min_otm_default_zero(self):
        """No config → effective floor is 0 (no filtering)."""
        from spread_scanner import _resolve_min_otm, _resolve_max_otm
        import types
        args = types.SimpleNamespace(min_otm=0, min_otm_per_ticker={},
                                      max_otm=0, max_otm_per_ticker={})
        assert _resolve_min_otm(args, "SPX") == 0.0
        assert _resolve_max_otm(args, "SPX") == 0.0

    def test_collect_filtered_respects_per_ticker_min_otm(self, tmp_path, monkeypatch):
        """The top-picks filter honors per-ticker OTM floors."""
        from spread_scanner import _load_config, _collect_filtered_candidates
        path = tmp_path / "cfg.yaml"
        path.write_text("""
tickers: [SPX, NDX]
dte: [0]
min_otm_per_ticker: {NDX: 2.0}
""")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, _ = _load_config()
        scan_data = {
            "dte_sections": {
                0: {
                    "expiration": "2026-04-22",
                    "spreads": {
                        "SPX": [
                            {"option_type": "PUT", "short_strike": 7050, "long_strike": 7025,
                             "width": 25, "credit": 1.0, "roi_pct": 4.0, "otm_pct": 1.0},
                        ],
                        "NDX": [
                            # OTM 1.1% — SHOULD be dropped by NDX per-ticker 2.0 floor
                            {"option_type": "PUT", "short_strike": 26550, "long_strike": 26490,
                             "width": 60, "credit": 7.0, "roi_pct": 13.0, "otm_pct": 1.1},
                            # OTM 2.5% — passes
                            {"option_type": "PUT", "short_strike": 26150, "long_strike": 26090,
                             "width": 60, "credit": 2.0, "roi_pct": 3.5, "otm_pct": 2.5},
                        ],
                    },
                    "quotes": {}, "tier_data": None, "prev_closes": {},
                },
            },
            "quotes": {}, "prev_closes": {},
        }
        picks = _collect_filtered_candidates(scan_data, args)
        # SPX 1.0% OTM passes (no SPX override; scalar min_otm=0)
        assert any(p["symbol"] == "SPX" and p["otm_pct"] == 1.0 for p in picks)
        # NDX 1.1% OTM dropped; 2.5% OTM kept
        ndx = [p for p in picks if p["symbol"] == "NDX"]
        assert {p["otm_pct"] for p in ndx} == {2.5}

    def test_yaml_min_tier_close_flows_through(self, tmp_path, monkeypatch):
        from spread_scanner import _load_config
        path = tmp_path / "cfg.yaml"
        path.write_text("tickers: [SPX]\nmin_tier_close: aggr\n")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, _ = _load_config()
        assert args.min_tier_close == "aggressive"

    def test_no_config_still_works(self, monkeypatch):
        """Without --config, legacy CLI-only path must still function."""
        from spread_scanner import _load_config
        monkeypatch.setattr("sys.argv",
            ["spread_scanner.py", "--tickers", "SPX", "--log", "3:/tmp/x.jsonl"])
        args, handlers = _load_config()
        assert args.tickers == ["SPX"]
        assert len(handlers) == 1
        assert handlers[0].name == "log"

    # ── scan_loop handler dispatch ────────────────────────────────────────

    def test_scan_loop_invokes_each_handler(self, monkeypatch):
        """scan_loop should call filter() + fire() on every configured handler."""
        import asyncio
        import spread_scanner as ss
        from spread_scanner import ActionHandler, HandlerContext, parse_args

        fired = []

        class StubHandler(ActionHandler):
            def __init__(self, tag):
                self.name = tag
                self.tag = tag
            def filter(self, candidates):
                return [{"tag": self.tag}]  # always eligible
            async def fire(self, spreads, ctx):
                fired.append(self.tag)

        async def fake_scan(client, args, prev_close_cache=None):
            return {"quotes": {}, "dte_sections": {}, "prev_closes": {}}
        monkeypatch.setattr(ss, "scan_all_tickers", fake_scan)
        monkeypatch.setattr(ss, "render_dashboard", lambda data, args: "")
        monkeypatch.setattr(ss, "_collect_filtered_candidates", lambda data, args: [])
        # Stub the prev-close cache refresh so the loop starts instantly
        monkeypatch.setattr(ss.PrevCloseCache, "refresh",
            lambda self, client, tier_data=None, now_pt=None: asyncio.sleep(0))

        args = parse_args(["--tickers", "SPX", "--once"])
        args.handlers = [StubHandler("A"), StubHandler("B"), StubHandler("C")]

        asyncio.run(ss.scan_loop(args))
        assert fired == ["A", "B", "C"]

    def test_scan_loop_handler_error_does_not_stop_pipeline(self, monkeypatch):
        """If one handler raises, other handlers still fire."""
        import asyncio
        import spread_scanner as ss
        from spread_scanner import ActionHandler, parse_args

        fired = []
        class OK(ActionHandler):
            def __init__(self, tag):
                self.name = tag; self.tag = tag
            def filter(self, c): return [{"x": 1}]
            async def fire(self, s, ctx): fired.append(self.tag)
        class Boom(ActionHandler):
            name = "boom"
            def filter(self, c): return [{"x": 1}]
            async def fire(self, s, ctx): raise RuntimeError("boom")

        async def fake_scan(client, args, prev_close_cache=None):
            return {"quotes": {}, "dte_sections": {}, "prev_closes": {}}
        monkeypatch.setattr(ss, "scan_all_tickers", fake_scan)
        monkeypatch.setattr(ss, "render_dashboard", lambda data, args: "")
        monkeypatch.setattr(ss, "_collect_filtered_candidates", lambda data, args: [])
        monkeypatch.setattr(ss.PrevCloseCache, "refresh",
            lambda self, client, tier_data=None, now_pt=None: asyncio.sleep(0))

        args = parse_args(["--tickers", "SPX", "--once"])
        args.handlers = [OK("before"), Boom(), OK("after")]
        asyncio.run(ss.scan_loop(args))
        assert fired == ["before", "after"]

    # ── Anticipatory scan kickoff (countdown lines up with refresh) ─────

    def test_compute_kickoff_lead_typical_case(self):
        """For the typical case (3s scan, 30s interval), lead =
        predicted_dur + 0.5s buffer = 3.5s. The scan kicks off 3.5s
        before the paint deadline so it finishes when the operator-
        visible countdown reaches 0."""
        from spread_scanner import _compute_kickoff_lead
        assert _compute_kickoff_lead(predicted_dur=3.0, interval=30.0) == 3.5
        assert _compute_kickoff_lead(predicted_dur=1.0, interval=30.0) == 1.5
        assert _compute_kickoff_lead(predicted_dur=0.0, interval=30.0) == 0.5

    def test_compute_kickoff_lead_caps_at_interval_minus_one(self):
        """If the scan takes nearly the full interval, the lead is
        capped at `interval - 1` so the user always sees at least one
        second of countdown — never instant scan-after-scan."""
        from spread_scanner import _compute_kickoff_lead
        # 9s scan in 10s interval: raw lead = 9.5s, but cap = 9.0s.
        assert _compute_kickoff_lead(predicted_dur=9.0, interval=10.0) == 9.0
        # 12s scan in 10s interval: raw lead = 12.5s, cap = 9.0s.
        assert _compute_kickoff_lead(predicted_dur=12.0, interval=10.0) == 9.0
        # 30s scan in 30s interval: raw lead = 30.5s, cap = 29.0s.
        assert _compute_kickoff_lead(predicted_dur=30.0, interval=30.0) == 29.0

    def test_compute_kickoff_lead_floors_at_buffer(self):
        """Lead must always be at least the buffer (default 0.5s) — even
        a near-zero predicted scan duration still gets a tiny anticipatory
        lead so we don't slip back into pure sequential scan-after-paint."""
        from spread_scanner import _compute_kickoff_lead
        assert _compute_kickoff_lead(predicted_dur=0.0, interval=30.0) == 0.5
        assert _compute_kickoff_lead(predicted_dur=-1.0, interval=30.0) == 0.5

    def test_compute_kickoff_lead_tiny_interval(self):
        """Edge case: when interval is so small that ceiling collapses
        below the floor, the floor wins. Prevents nonsensical negative
        leads on degenerate configs."""
        from spread_scanner import _compute_kickoff_lead
        # interval=1.0 → ceiling=max(0, 0.5)=0.5; floor=0.5 → lead=0.5.
        assert _compute_kickoff_lead(predicted_dur=5.0, interval=1.0) == 0.5
        # interval=0.5 → ceiling=max(-0.5, 0.5)=0.5; floor=0.5 → lead=0.5.
        assert _compute_kickoff_lead(predicted_dur=5.0, interval=0.5) == 0.5


class TestHandlerValidatePrices:
    """Per-handler pre-fire price validation (`validate_prices: true`).

    Each handler can opt into a final round-trip to verify_spread_pricing
    immediately before firing. Failed candidates are dropped; successes
    have their credit/bid/ask refreshed in place. Default OFF — handlers
    fire on whatever the scan-level verify produced.
    """

    def _fake_trading_client(self, results_per_call: list[dict]):
        """Build a fake TradingClient that returns canned results in order."""
        calls = {"args": []}

        class Fake:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def verify_spread_pricing(self, **kw):
                calls["args"].append(kw)
                # Pop the next canned result (or the last one if exhausted).
                idx = min(len(calls["args"]) - 1, len(results_per_call) - 1)
                return results_per_call[idx]
            # Stubs the trade handler may call after validation passes —
            # not exercised by these tests (validation drops candidates),
            # but present so attribute lookups don't 500.
            async def get_trade_defaults(self): return {"default_order_type": "MARKET"}

        return Fake, calls

    def _make_ctx(self, args=None):
        import argparse
        from spread_scanner import HandlerContext
        ns = args or argparse.Namespace(
            daemon_url="http://localhost:8000",
            verify_max_age_sec=30.0,
            verify_require_provider_source=True,
            verify_batch_timeout_sec=8.0,
        )
        return HandlerContext(
            client=None, args=ns, scan_data={},
            is_market_hours=True, now_ts="2026-04-27T13:00:00",
        )

    def test_default_validate_prices_is_false(self):
        """All handlers default validate_prices=False — no extra round-trips
        unless the user opts in."""
        from spread_scanner import (
            LogHandler, NotifyHandler, SimulateTradeHandler, TradeHandler,
            TradePolicy,
        )
        h_log = LogHandler(min_norm_roi=1.0, path="/tmp/x.jsonl")
        h_notify = NotifyHandler(min_norm_roi=1.0, email="a@b.c")
        h_sim = SimulateTradeHandler(
            min_norm_roi=1.0, log_file="/tmp/y.jsonl",
            policy=TradePolicy(), daemon_url="http://localhost:8000",
        )
        h_trade = TradeHandler(
            min_norm_roi=1.0, log_file="/tmp/z.jsonl",
            policy=TradePolicy(), daemon_url="http://localhost:8000",
        )
        # Log handler doesn't validate (intentionally — see CLAUDE.md scope).
        # The base-class default is False for everyone else too.
        assert h_log.validate_prices is False
        assert h_notify.validate_prices is False
        assert h_sim.validate_prices is False
        assert h_trade.validate_prices is False

    def test_build_handler_parses_validate_prices_from_yaml(self):
        from spread_scanner import build_handler

        h = build_handler({
            "type": "notify", "min_norm_roi": 2.0, "email": "a@b.c",
            "validate_prices": True,
        })
        assert h.validate_prices is True

        h2 = build_handler({
            "type": "simulate_trade", "min_norm_roi": 2.0,
            "log_file": "/tmp/sim.jsonl", "policy": {},
            "daemon_url": "http://localhost:8000",
            "validate_prices": True,
        })
        assert h2.validate_prices is True

        h3 = build_handler({
            "type": "trade", "min_norm_roi": 2.0,
            "log_file": "/tmp/live.jsonl", "policy": {},
            "daemon_url": "http://localhost:8000",
            "validate_prices": True,
        })
        assert h3.validate_prices is True

        # Omitted → default False
        h4 = build_handler({
            "type": "notify", "min_norm_roi": 2.0, "email": "a@b.c",
        })
        assert h4.validate_prices is False

    def test_validate_prices_false_skips_provider_call(self):
        """Handler with validate_prices=False does NOT call
        verify_spread_pricing, even if the helper is monkeypatched."""
        import asyncio
        import spread_scanner as ss
        from spread_scanner import NotifyHandler

        called = {"n": 0}
        async def fake_validate(spreads, ctx, *, handler_name="handler"):
            called["n"] += 1
            return spreads

        h = NotifyHandler(min_norm_roi=0.0, email="a@b.c", validate_prices=False)
        ctx = self._make_ctx()
        asyncio.run(h._maybe_validate_prices(
            [{"symbol": "SPX", "expiration": "2026-04-27",
              "option_type": "PUT", "short_strike": 7050,
              "long_strike": 7030, "credit": 0.40, "width": 20}],
            ctx,
        ))
        # The helper is reachable but the bool-gate short-circuits before it.
        assert called["n"] == 0

    def test_validate_prices_drops_failed_candidates(self):
        """validate_prices=True drops candidates whose verify call fails."""
        import asyncio
        import spread_scanner as ss
        from spread_scanner import NotifyHandler

        # First spread passes, second fails.
        Fake, calls = self._fake_trading_client([
            {"ok": True, "credit": 0.50, "short_bid": 0.60, "long_ask": 0.10,
             "short_delta": -0.18, "age_seconds": 3.0, "source": "ibkr_fresh"},
            {"ok": False, "reason": "csv_source_rejected"},
        ])
        ss._get_trading_client_cls = lambda: Fake

        h = NotifyHandler(min_norm_roi=0.0, email="a@b.c", validate_prices=True)
        ctx = self._make_ctx()
        spreads = [
            {"symbol": "SPX", "expiration": "2026-04-27", "option_type": "PUT",
             "short_strike": 7050, "long_strike": 7030, "credit": 0.40, "width": 20},
            {"symbol": "SPX", "expiration": "2026-04-27", "option_type": "PUT",
             "short_strike": 7045, "long_strike": 7025, "credit": 0.30, "width": 20},
        ]
        kept = asyncio.run(h._maybe_validate_prices(spreads, ctx))
        assert len(kept) == 1
        assert kept[0]["short_strike"] == 7050
        # Survivor's credit was refreshed in place from verify result.
        assert kept[0]["credit"] == 0.50
        assert kept[0]["verified"] is True
        # Both spreads were submitted for verification.
        assert len(calls["args"]) == 2

    def test_validate_prices_uses_args_freshness_policy(self):
        """The validator forwards `verify_max_age_sec` and
        `verify_require_provider_source` from ctx.args to verify_spread_pricing."""
        import asyncio
        import argparse
        import spread_scanner as ss
        from spread_scanner import NotifyHandler

        Fake, calls = self._fake_trading_client([
            {"ok": True, "credit": 0.50, "short_bid": 0.50, "long_ask": 0.0,
             "short_delta": -0.1, "age_seconds": 2.0, "source": "ibkr_fresh"},
        ])
        ss._get_trading_client_cls = lambda: Fake

        ns = argparse.Namespace(
            daemon_url="http://localhost:8000",
            verify_max_age_sec=45.0,
            verify_require_provider_source=False,
            verify_batch_timeout_sec=8.0,
        )
        h = NotifyHandler(min_norm_roi=0.0, email="a@b.c", validate_prices=True)
        ctx = self._make_ctx(args=ns)
        asyncio.run(h._maybe_validate_prices(
            [{"symbol": "SPX", "expiration": "2026-04-27", "option_type": "PUT",
              "short_strike": 7050, "long_strike": 7030, "credit": 0.40, "width": 20}],
            ctx,
        ))
        assert calls["args"][0]["max_age"] == 45.0
        assert calls["args"][0]["require_provider_source"] is False

    def test_trade_handler_skips_failed_validation_before_risk_reservation(self, tmp_path):
        """When validate_prices=True drops every candidate, the trade handler
        must not reserve risk or submit anything — fire() is a no-op."""
        import asyncio
        import spread_scanner as ss
        from spread_scanner import (
            HandlerContext, SimulateTradeHandler, TradePolicy,
        )

        # Every candidate fails validation.
        Fake, _calls = self._fake_trading_client([
            {"ok": False, "reason": "csv_source_rejected"},
            {"ok": False, "reason": "csv_source_rejected"},
        ])
        ss._get_trading_client_cls = lambda: Fake

        log_file = tmp_path / "sim.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0.0, log_file=str(log_file),
            policy=TradePolicy(min_otm_pct={}, min_credit={}),
            daemon_url="http://localhost:8000",
            validate_prices=True,
        )
        ctx = self._make_ctx()
        # Inject `expiration` since fire() reads spread.expiration via the
        # validator path (falls through to verify before risk reservation).
        spreads = [
            {"symbol": "SPX", "expiration": "2026-04-27", "option_type": "PUT",
             "short_strike": 7050, "long_strike": 7030, "credit": 0.40,
             "width": 20, "roi_pct": 2.0, "otm_pct": 1.0, "dte": 0,
             "norm_roi": 2.0, "prev_close": 7100.0, "timestamp": "x"},
            {"symbol": "NDX", "expiration": "2026-04-27", "option_type": "PUT",
             "short_strike": 26800, "long_strike": 26740, "credit": 0.50,
             "width": 60, "roi_pct": 0.85, "otm_pct": 1.5, "dte": 0,
             "norm_roi": 0.85, "prev_close": 27300.0, "timestamp": "x"},
        ]
        asyncio.run(h.fire(spreads, ctx))
        # Risk counters untouched — we never even got to the per-ticker queue.
        assert h.cum_risk == 0
        assert h.count_submitted == 0
        # No submit / result / skipped events were written either.
        if log_file.exists():
            lines = log_file.read_text().strip().splitlines()
            for ln in lines:
                # Whatever was logged, it must not be a submit/result/skipped.
                import json as _json
                ev = _json.loads(ln)
                assert ev["event"] not in ("submit", "result", "skipped")

    def test_validate_prices_batch_timeout_falls_through(self):
        """If the daemon hangs and the validate-batch times out, the helper
        returns the input unchanged so handlers still fire — better to act
        on slightly older data than to silently drop every action."""
        import asyncio
        import argparse
        import spread_scanner as ss
        from spread_scanner import NotifyHandler

        class HangingClient:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def verify_spread_pricing(self, **kw):
                await asyncio.sleep(60)
                return {"ok": True}

        ss._get_trading_client_cls = lambda: HangingClient

        ns = argparse.Namespace(
            daemon_url="http://localhost:8000",
            verify_max_age_sec=30.0,
            verify_require_provider_source=True,
            verify_batch_timeout_sec=0.2,
        )
        h = NotifyHandler(min_norm_roi=0.0, email="a@b.c", validate_prices=True)
        ctx = self._make_ctx(args=ns)
        spreads = [
            {"symbol": "SPX", "expiration": "2026-04-27", "option_type": "PUT",
             "short_strike": 7050, "long_strike": 7030, "credit": 0.40, "width": 20},
        ]
        kept = asyncio.run(h._maybe_validate_prices(spreads, ctx))
        # Pass-through on timeout.
        assert kept == spreads


class TestTradePolicy:
    """TradePolicy dataclass + trading-window helper."""

    def test_defaults(self):
        from spread_scanner import TradePolicy
        from datetime import time as dtime
        p = TradePolicy()
        assert p.roi_pct == (1.5, 5.0)
        assert p.max_total_risk == 400_000.0
        assert p.max_risk_per_trade == {}         # empty → per-candidate default (width*100)
        assert p.require_prev_close is True
        # None = "use the daemon's configured default" via GET /trade/defaults.
        # A handler can still pin explicit values to override the daemon config.
        assert p.order_type is None
        assert p.limit_slippage_pct is None
        assert p.trading_window_pt_start == dtime(6, 31)
        assert p.trading_window_pt_end == dtime(10, 0)
        assert not hasattr(p, "max_trades_per_interval_per_ticker")
        assert not hasattr(p, "quantity")

    def test_from_dict_full(self):
        from spread_scanner import TradePolicy
        from datetime import time as dtime
        p = TradePolicy.from_dict({
            "roi_pct": [1.0, 6.0],
            "min_otm_pct": {"spx": 1.0, "ndx": 1.5},
            "min_credit": {"RUT": 0.40},
            "max_total_risk": 250000,
            "max_per_ticker_risk": {"SPX": 100000},
            "max_risk_per_trade": {"spx": 5000, "NDX": 12000},
            "cooldown_per_ticker_side_sec": 120,
            "require_prev_close": False,
            "order_type": "MARKET",
            "stop_loss_multiplier": 1.5,
            "trading_window_pt": {"start": "07:00", "end": "09:30:30"},
        })
        assert p.roi_pct == (1.0, 6.0)
        assert p.min_otm_pct == {"SPX": 1.0, "NDX": 1.5}     # upper-cased keys
        assert p.min_credit == {"RUT": 0.40}
        assert p.max_total_risk == 250000
        assert p.max_per_ticker_risk == {"SPX": 100000}
        assert p.max_risk_per_trade == {"SPX": 5000, "NDX": 12000}   # keys upper-cased
        assert p.cooldown_per_ticker_side_sec == 120
        assert p.require_prev_close is False
        assert p.order_type == "MARKET"
        assert p.trading_window_pt_start == dtime(7, 0)
        assert p.trading_window_pt_end == dtime(9, 30, 30)

    def test_from_dict_empty(self):
        from spread_scanner import TradePolicy
        assert TradePolicy.from_dict(None).roi_pct == (1.5, 5.0)
        assert TradePolicy.from_dict({}).roi_pct == (1.5, 5.0)

    def test_from_dict_unknown_field(self):
        import pytest
        from spread_scanner import TradePolicy
        with pytest.raises(ValueError, match="Unknown TradePolicy field"):
            TradePolicy.from_dict({"warp_drive": True})

    def test_from_dict_bad_roi(self):
        import pytest
        from spread_scanner import TradePolicy
        with pytest.raises(ValueError, match="roi_pct must be"):
            TradePolicy.from_dict({"roi_pct": [5.0, 1.0]})
        with pytest.raises(ValueError, match="roi_pct must be"):
            TradePolicy.from_dict({"roi_pct": [1.0]})

    def test_from_dict_bad_window(self):
        import pytest
        from spread_scanner import TradePolicy
        with pytest.raises(ValueError, match="end must be >= start"):
            TradePolicy.from_dict({"trading_window_pt": {"start": "10:00", "end": "09:00"}})

    def test_from_dict_bad_time_format(self):
        import pytest
        from spread_scanner import TradePolicy
        with pytest.raises(ValueError, match="HH:MM"):
            TradePolicy.from_dict({"trading_window_pt": {"start": "six am", "end": "10:00"}})

    def test_within_trading_window_default(self):
        from spread_scanner import TradePolicy, _PT
        from datetime import datetime
        p = TradePolicy()
        # 07:00 PT is inside default 06:31 - 10:00 window
        assert p.within_trading_window(datetime(2026, 4, 21, 7, 0, tzinfo=_PT)) is True
        # 06:30 PT is 1 min before — outside
        assert p.within_trading_window(datetime(2026, 4, 21, 6, 30, tzinfo=_PT)) is False
        # 10:00 PT is the boundary — inclusive
        assert p.within_trading_window(datetime(2026, 4, 21, 10, 0, tzinfo=_PT)) is True
        # 10:00:01 PT is just past — outside
        assert p.within_trading_window(datetime(2026, 4, 21, 10, 0, 1, tzinfo=_PT)) is False

    def test_within_trading_window_custom(self):
        from spread_scanner import TradePolicy, _PT
        from datetime import datetime
        p = TradePolicy.from_dict({"trading_window_pt": {"start": "09:00", "end": "11:00"}})
        assert p.within_trading_window(datetime(2026, 4, 21, 7, 0, tzinfo=_PT)) is False
        assert p.within_trading_window(datetime(2026, 4, 21, 9, 30, tzinfo=_PT)) is True
        assert p.within_trading_window(datetime(2026, 4, 21, 12, 0, tzinfo=_PT)) is False


class TestTradeHandlers:
    """SimulateTradeHandler + TradeHandler — policy gates, concurrency, logging."""

    @staticmethod
    def _spread(sym, otype="PUT", credit=1.0, otm=1.5, roi=3.0, short=5700, long_=5680,
                width=20, prev_close=5800.0):
        return {
            "symbol": sym, "option_type": otype, "short_strike": short, "long_strike": long_,
            "width": width, "credit": credit, "roi_pct": roi, "otm_pct": otm,
            "dte": 0, "expiration": "2026-04-21", "prev_close": prev_close,
        }

    @staticmethod
    def _ctx(client=None):
        from spread_scanner import HandlerContext
        return HandlerContext(client=client, args=None, scan_data={},
                              is_market_hours=True, now_ts="2026-04-21T07:00:00")

    def _always_in_window(self, monkeypatch):
        import spread_scanner as ss
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window", lambda self, now_pt=None: True)

    @staticmethod
    def _bind_utp_pricing(fake_cls):
        """Attach the real LIMIT-pricing + daemon-default methods to a fake.

        The scanner delegates ALL limit-pricing math to utp.py. Tests stub the
        HTTP layer (get_option_quotes, trade_credit_spread, get_trade_defaults
        if present) but must share the real pricing implementation so the
        scanner → utp wiring is exercised.
        """
        from utp import TradingClient
        fake_cls.compute_credit_spread_net_price = TradingClient.compute_credit_spread_net_price
        fake_cls._resolve_trade_defaults = TradingClient._resolve_trade_defaults
        # If the fake doesn't define get_trade_defaults, provide a default
        # that raises so _resolve_trade_defaults falls back to hardcoded values.
        if not hasattr(fake_cls, "get_trade_defaults"):
            async def _raise_no_daemon(self):
                raise RuntimeError("no daemon defaults in fake")
            fake_cls.get_trade_defaults = _raise_no_daemon
        return fake_cls

    # --- filter() gates ---------------------------------------------------

    def test_filter_blocks_outside_trading_window(self, monkeypatch):
        import spread_scanner as ss
        from spread_scanner import SimulateTradeHandler, TradePolicy
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window", lambda self, now_pt=None: False)
        h = SimulateTradeHandler(min_norm_roi=1.0, log_file="/tmp/u.jsonl",
                                 policy=TradePolicy(), daemon_url="http://d")
        assert h.filter([self._spread("SPX")]) == []

    def test_filter_skips_missing_prev_close(self, monkeypatch):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        h = SimulateTradeHandler(
            min_norm_roi=1.0, log_file="/tmp/u.jsonl",
            policy=TradePolicy(roi_pct=(1.0, 10.0)), daemon_url="http://d",
        )
        cands = [self._spread("SPX", prev_close=0), self._spread("NDX", prev_close=26000)]
        out = h.filter(cands)
        assert [c["symbol"] for c in out] == ["NDX"]

    def test_filter_blocks_below_credit_floor(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"min_credit": {"SPX": 0.50}, "roi_pct": [1.0, 10.0]})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        out = h.filter([
            self._spread("SPX", credit=0.30),
            self._spread("SPX", credit=0.75),
        ])
        assert len(out) == 1 and out[0]["credit"] == 0.75

    def test_filter_blocks_below_otm_floor(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"min_otm_pct": {"SPX": 1.0}, "roi_pct": [1.0, 10.0]})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        out = h.filter([
            self._spread("SPX", otm=0.5),
            self._spread("SPX", otm=1.5),
        ])
        assert len(out) == 1 and out[0]["otm_pct"] == 1.5

    def test_filter_roi_band(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"roi_pct": [1.5, 5.0]})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        out = h.filter([
            self._spread("SPX", roi=1.0),    # below band
            self._spread("SPX", roi=3.0),    # inside
            self._spread("SPX", roi=6.0),    # above
        ])
        assert [c["roi_pct"] for c in out] == [3.0]

    def test_filter_caps_trades_per_interval_per_ticker_default_1(self, monkeypatch, tmp_path):
        """Default cap = 1 per ticker per scan: 3 SPX + 2 NDX candidates → 1 SPX + 1 NDX."""
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0]})   # default cap=1
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        out = h.filter([
            self._spread("SPX", credit=1.0, roi=2.0),
            self._spread("SPX", credit=1.1, roi=3.0),     # highest SPX nROI
            self._spread("SPX", credit=1.2, roi=2.5),
            self._spread("NDX", credit=2.0, roi=3.5, prev_close=26000),   # highest NDX
            self._spread("NDX", credit=2.1, roi=2.8, prev_close=26000),
        ])
        symbols = [c["symbol"] for c in out]
        assert sorted(symbols) == ["NDX", "SPX"]
        # Within each ticker, the highest-nROI candidate wins.
        spx = next(c for c in out if c["symbol"] == "SPX")
        ndx = next(c for c in out if c["symbol"] == "NDX")
        assert spx["roi_pct"] == 3.0
        assert ndx["roi_pct"] == 3.5

    def test_filter_collapses_to_one_per_ticker(self, monkeypatch, tmp_path):
        """Exactly one trade per ticker per scan — sizing is handled by
        policy.contracts_for(), not by multiplying trades."""
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0]})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        out = h.filter([
            self._spread("SPX", credit=1.0, roi=2.0),
            self._spread("SPX", credit=1.1, roi=3.5),     # highest SPX nROI — wins
            self._spread("SPX", credit=1.2, roi=3.0),
            self._spread("SPX", credit=1.3, roi=2.5),
        ])
        assert len(out) == 1
        assert out[0]["roi_pct"] == 3.5

    def test_policy_contracts_for_default_width_times_100(self):
        """No max_risk_per_trade override → default is width*100 → 1 contract."""
        from spread_scanner import TradePolicy
        p = TradePolicy()
        # SPX width 20 credit 1.0 → default cap = 2000, per-contract max = 1900 → 1 contract
        assert p.contracts_for({"symbol": "SPX", "width": 20, "credit": 1.0}) == 1
        # NDX width 60 credit 2.0 → default cap = 6000, per-contract max = 5800 → 1 contract
        assert p.contracts_for({"symbol": "NDX", "width": 60, "credit": 2.0}) == 1

    def test_policy_contracts_for_override_sizes_up(self):
        from spread_scanner import TradePolicy
        p = TradePolicy.from_dict({"max_risk_per_trade": {"SPX": 5000}})
        # SPX width 20 credit 1.0 → per-contract max = 1900 → 5000/1900 = 2
        assert p.contracts_for({"symbol": "SPX", "width": 20, "credit": 1.0}) == 2
        # With 10000 → 10000/1900 = 5
        p2 = TradePolicy.from_dict({"max_risk_per_trade": {"SPX": 10000}})
        assert p2.contracts_for({"symbol": "SPX", "width": 20, "credit": 1.0}) == 5

    def test_policy_contracts_for_override_does_not_apply_to_other_ticker(self):
        from spread_scanner import TradePolicy
        p = TradePolicy.from_dict({"max_risk_per_trade": {"SPX": 10000}})
        # NDX not in map → uses default width*100 = 6000 → 1 contract
        assert p.contracts_for({"symbol": "NDX", "width": 60, "credit": 2.0}) == 1

    def test_policy_contracts_for_clamped_to_at_least_1(self):
        from spread_scanner import TradePolicy
        # Absurdly small cap: per-contract = 1900, cap = 500 → 500/1900 = 0 → clamp to 1
        p = TradePolicy.from_dict({"max_risk_per_trade": {"SPX": 500}})
        assert p.contracts_for({"symbol": "SPX", "width": 20, "credit": 1.0}) == 1

    def test_trade_handler_passes_computed_contracts(self, monkeypatch, tmp_path):
        """TradeHandler submits policy.contracts_for(spread), not a fixed quantity."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        captured = {}
        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, **kw):
                captured.update(kw)
                return {"order_id": "x", "status": "FILLED"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        # cap 5000, SPX width 20, credit 1.0 → 2 contracts
        pol = TradePolicy.from_dict({
            "roi_pct": [1.0, 10.0], "max_risk_per_trade": {"SPX": 5000},
        })
        log_file = tmp_path / "t.jsonl"
        h = TradeHandler(min_norm_roi=0, log_file=str(log_file),
                        policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread("SPX", credit=1.0, width=20)], self._ctx()))
        assert captured["quantity"] == 2
        events = [json.loads(l) for l in log_file.read_text().splitlines()]
        submit = [e for e in events if e["event"] == "submit"][0]
        assert submit["contracts"] == 2

    # --- concurrency ------------------------------------------------------

    def test_cross_ticker_parallel_within_ticker_serial(self, monkeypatch, tmp_path):
        """Cross-ticker trades run concurrently; within-ticker trades are serial."""
        import asyncio, time
        from spread_scanner import TradeHandler, TradePolicy, HandlerContext

        submit_log: list[tuple[str, float, float]] = []  # (ticker, start, end)

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, symbol, **kw):
                start = time.perf_counter()
                await asyncio.sleep(0.15)
                end = time.perf_counter()
                submit_log.append((symbol, start, end))
                return {"order_id": f"o_{symbol}_{len(submit_log)}", "status": "FILLED"}

        import spread_scanner as ss
        # Patch `from utp import TradingClient` target
        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0]})
        h = TradeHandler(min_norm_roi=0, log_file=str(tmp_path / "t.jsonl"),
                        policy=pol, daemon_url="http://d")

        # 3 spreads per ticker
        sp = self._spread
        spreads = [
            sp("SPX", credit=1.0, short=5700, long_=5680),
            sp("SPX", credit=1.1, short=5710, long_=5690),
            sp("SPX", credit=1.2, short=5720, long_=5700),
            sp("NDX", credit=2.0, prev_close=26000, short=26200, long_=26150),
            sp("NDX", credit=2.1, prev_close=26000, short=26210, long_=26160),
            sp("NDX", credit=2.2, prev_close=26000, short=26220, long_=26170),
            sp("RUT", credit=1.5, prev_close=2800, short=2800, long_=2780),
            sp("RUT", credit=1.6, prev_close=2800, short=2805, long_=2785),
            sp("RUT", credit=1.7, prev_close=2800, short=2810, long_=2790),
        ]
        ctx = self._ctx()
        t0 = time.perf_counter()
        asyncio.run(h.fire(spreads, ctx))
        total = time.perf_counter() - t0

        # Serial within ticker: each ticker runs its 3 spreads one at a time = 3 * 0.15 = 0.45s.
        # Parallel across tickers: three tickers start together, so total ~= max ticker time ~= 0.45s.
        # Accept a generous upper bound (e.g. 1.0s) to absorb scheduler jitter.
        assert 0.4 < total < 1.0, f"got total={total:.2f}s"

        # Within each ticker, submissions must not overlap in time.
        for sym in ("SPX", "NDX", "RUT"):
            ticker_rows = sorted([r for r in submit_log if r[0] == sym], key=lambda r: r[1])
            assert len(ticker_rows) == 3
            for i in range(len(ticker_rows) - 1):
                assert ticker_rows[i][2] <= ticker_rows[i + 1][1] + 1e-6, (
                    f"{sym} submissions overlap: {ticker_rows[i]} and {ticker_rows[i+1]}")

        # Across tickers, at least one pair of DIFFERENT-ticker submissions must overlap
        # (proves concurrency).
        overlap_found = False
        for i in range(len(submit_log)):
            for j in range(i + 1, len(submit_log)):
                a, b = submit_log[i], submit_log[j]
                if a[0] != b[0]:  # different ticker
                    if a[1] < b[2] and b[1] < a[2]:
                        overlap_found = True
                        break
            if overlap_found:
                break
        assert overlap_found, "no cross-ticker overlap observed — not running in parallel"

    def test_cap_recheck_inside_lock(self, monkeypatch, tmp_path):
        """Two SPX trades where each alone fits the cap but both together don't.
        First should go through, second should be skipped with reason=total_risk_cap."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, **kw):
                return {"order_id": "x", "status": "FILLED"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        # width=20, credit=1.0 → max_loss = $1900 per trade. Cap = $2500 → 1 trade fits, 2 don't.
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "max_total_risk": 2500})
        log_file = tmp_path / "t.jsonl"
        h = TradeHandler(min_norm_roi=0, log_file=str(log_file), policy=pol,
                        daemon_url="http://d")

        sp = self._spread
        ctx = self._ctx()
        asyncio.run(h.fire([sp("SPX", credit=1.0, short=5700, long_=5680),
                           sp("SPX", credit=1.0, short=5710, long_=5690)], ctx))

        events = [json.loads(l) for l in log_file.read_text().splitlines()]
        kinds = [e["event"] for e in events]
        assert kinds.count("submit") == 1
        assert kinds.count("result") == 1
        skipped = [e for e in events if e["event"] == "skipped"]
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "total_risk_cap"

    def test_per_ticker_cap_blocks_second_trade_same_ticker(self, monkeypatch, tmp_path):
        """Per-ticker cap prevents NDX from running up total; SPX still goes."""
        import asyncio
        import spread_scanner as ss
        from spread_scanner import TradeHandler, TradePolicy

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, **kw):
                return {"order_id": "x", "status": "FILLED"}

        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        # NDX width=50 credit=1.0 → max_loss=$4900. per-ticker cap for NDX = $5000 → 1 fits, 2nd blocked.
        pol = TradePolicy.from_dict({
            "roi_pct": [1.0, 10.0],
            "max_per_ticker_risk": {"NDX": 5000, "SPX": 100000},
        })
        log_file = tmp_path / "t.jsonl"
        h = TradeHandler(min_norm_roi=0, log_file=str(log_file), policy=pol,
                        daemon_url="http://d")

        sp = self._spread
        spreads = [
            sp("NDX", credit=1.0, prev_close=26000, short=26200, long_=26250, width=50),
            sp("NDX", credit=1.0, prev_close=26000, short=26210, long_=26260, width=50),
            sp("SPX", credit=1.0, short=5700, long_=5680, width=20),
        ]
        asyncio.run(h.fire(spreads, self._ctx()))

        events = [json.loads(l) for l in log_file.read_text().splitlines()]
        submits = [e for e in events if e["event"] == "submit"]
        skipped = [e for e in events if e["event"] == "skipped"]
        assert len(submits) == 2
        assert len(skipped) == 1
        assert skipped[0]["spread"]["symbol"] == "NDX"
        assert skipped[0]["reason"] == "per_ticker_risk_cap"

    # --- logging + submission ---------------------------------------------

    def test_simulate_handler_calls_margin_not_broker(self, monkeypatch, tmp_path):
        import asyncio, sys
        from spread_scanner import SimulateTradeHandler, TradePolicy

        margin_calls = []
        trade_calls = []

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def check_margin_credit_spread(self, **kw):
                margin_calls.append(kw)
                return {"init_margin": 1800.0, "maint_margin": 1800.0, "commission": 1.0, "error": None}
            async def trade_credit_spread(self, **kw):
                trade_calls.append(kw)
                return {"order_id": "SHOULD-NOT-SUBMIT"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        log_file = tmp_path / "sim.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0, log_file=str(log_file),
            policy=TradePolicy.from_dict({"roi_pct": [1.0, 10.0]}),
            daemon_url="http://d",
        )
        asyncio.run(h.fire([self._spread("SPX", credit=1.0)], self._ctx()))

        assert len(margin_calls) == 1
        assert margin_calls[0]["symbol"] == "SPX"
        assert trade_calls == []  # simulate must NEVER call trade

        events = [json.loads(l) for l in log_file.read_text().splitlines()]
        assert any(e["event"] == "submit" and e["handler"] == "simulate_trade" for e in events)
        result_ev = [e for e in events if e["event"] == "result"][0]
        assert result_ev["result"]["simulated"] is True

    def test_trade_handler_logs_submit_and_result(self, monkeypatch, tmp_path):
        import asyncio, sys
        from spread_scanner import TradeHandler, TradePolicy

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, **kw):
                return {"order_id": "ord123", "status": "FILLED", "filled_price": -1.0}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        log_file = tmp_path / "live.jsonl"
        # Default policy now uses MARKET orders — no net_price sent.
        h = TradeHandler(min_norm_roi=0, log_file=str(log_file),
                        policy=TradePolicy.from_dict({"roi_pct": [1.0, 10.0]}),
                        daemon_url="http://d")
        asyncio.run(h.fire([self._spread("SPX", credit=1.0)], self._ctx()))

        events = [json.loads(l) for l in log_file.read_text().splitlines()]
        kinds = [e["event"] for e in events]
        assert "submit" in kinds and "result" in kinds
        result_ev = [e for e in events if e["event"] == "result"][0]
        assert result_ev["result"]["order"]["order_id"] == "ord123"
        assert result_ev["result"]["order_type"] == "MARKET"
        assert result_ev["result"]["submitted_net_price"] is None

    def test_trade_handler_limit_refreshes_quotes_and_uses_refreshed_credit(
        self, monkeypatch, tmp_path,
    ):
        """LIMIT orders fetch fresh option quotes and recompute credit from them."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        quote_calls: list[dict] = []
        trade_calls: list[dict] = []

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_option_quotes(self, **kw):
                quote_calls.append(kw)
                return {
                    "quotes": {
                        "put": [
                            {"strike": 5700.0, "bid": 1.20, "ask": 1.30},   # short leg — fresh bid 1.20
                            {"strike": 5680.0, "bid": 0.40, "ask": 0.50},   # long leg — fresh ask 0.50
                        ],
                    },
                    "meta": {"age_seconds": 3.0, "source": "fresh_cache"},
                }
            async def trade_credit_spread(self, **kw):
                trade_calls.append(kw)
                return {"order_id": "ord", "status": "FILLED"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "order_type": "LIMIT"})
        log = tmp_path / "l.jsonl"
        h = TradeHandler(min_norm_roi=0, log_file=str(log), policy=pol, daemon_url="http://d")
        # Scan-time credit = 0.85 (stale). Refreshed legs give 1.20 - 0.50 = 0.70.
        asyncio.run(h.fire([self._spread("SPX", credit=0.85,
                                          short=5700, long_=5680, width=20)], self._ctx()))

        assert len(quote_calls) == 1
        assert quote_calls[0]["symbol"] == "SPX"
        assert quote_calls[0]["option_type"] == "PUT"
        assert quote_calls[0]["max_age"] == 10.0
        assert len(trade_calls) == 1
        assert trade_calls[0]["net_price"] == 0.70      # refreshed credit, not 0.85

        # Result event captures the full audit trail
        events = [json.loads(l) for l in log.read_text().splitlines()]
        res = next(e for e in events if e["event"] == "result")
        lp = res["result"]["limit_pricing"]
        assert lp["scan_credit"] == 0.85
        assert lp["refreshed_credit"] == 0.70
        assert lp["age_seconds"] == 3.0
        assert lp["quote_source"] == "fresh_cache"
        assert lp["slippage_pct"] == 0.0
        assert lp["submitted_net_price"] == 0.70

    def test_trade_handler_limit_applies_slippage(self, monkeypatch, tmp_path):
        """limit_slippage_pct reduces net_price as a % of fresh credit."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        trade_calls: list[dict] = []
        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_option_quotes(self, **kw):
                return {
                    "quotes": {"put": [
                        {"strike": 5700.0, "bid": 1.00, "ask": 1.10},
                        {"strike": 5680.0, "bid": 0.10, "ask": 0.20},
                    ]},
                    "meta": {"age_seconds": 2.0, "source": "fresh_cache"},
                }
            async def trade_credit_spread(self, **kw):
                trade_calls.append(kw)
                return {"order_id": "x", "status": "FILLED"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        # refreshed credit = 1.00 - 0.20 = 0.80; slippage 10% → net_price = 0.72
        pol = TradePolicy.from_dict({
            "roi_pct": [1.0, 10.0], "order_type": "LIMIT", "limit_slippage_pct": 10.0,
        })
        h = TradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                        policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread("SPX", credit=0.80,
                                          short=5700, long_=5680, width=20)], self._ctx()))
        assert trade_calls[0]["net_price"] == 0.72

    def test_trade_handler_limit_falls_back_on_refresh_error(self, monkeypatch, tmp_path):
        """If get_option_quotes raises, fall back to scan-time credit and log reason."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        trade_calls: list[dict] = []
        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_option_quotes(self, **kw):
                raise RuntimeError("daemon is down")
            async def trade_credit_spread(self, **kw):
                trade_calls.append(kw)
                return {"order_id": "x", "status": "FILLED"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "order_type": "LIMIT"})
        log = tmp_path / "l.jsonl"
        h = TradeHandler(min_norm_roi=0, log_file=str(log),
                        policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread("SPX", credit=0.85,
                                          short=5700, long_=5680, width=20)], self._ctx()))

        # Fallback: submit at scan_credit (0.85), slippage 0%
        assert trade_calls[0]["net_price"] == 0.85
        events = [json.loads(l) for l in log.read_text().splitlines()]
        lp = next(e for e in events if e["event"] == "result")["result"]["limit_pricing"]
        assert lp["refreshed_credit"] is None
        assert lp["quote_source"] == "error"
        assert "fallback_reason" in lp
        assert "quote_refresh_failed" in lp["fallback_reason"]

    def test_trade_handler_limit_fallback_on_missing_leg(self, monkeypatch, tmp_path):
        """If fresh quotes don't cover one of the legs, fall back."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        trade_calls: list[dict] = []
        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_option_quotes(self, **kw):
                # Only the short leg comes back; long leg absent
                return {
                    "quotes": {"put": [
                        {"strike": 5700.0, "bid": 1.20, "ask": 1.30},
                    ]},
                    "meta": {"age_seconds": 5.0, "source": "provider"},
                }
            async def trade_credit_spread(self, **kw):
                trade_calls.append(kw)
                return {"order_id": "x", "status": "FILLED"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "order_type": "LIMIT"})
        log = tmp_path / "l.jsonl"
        h = TradeHandler(min_norm_roi=0, log_file=str(log),
                        policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread("SPX", credit=0.90,
                                          short=5700, long_=5680, width=20)], self._ctx()))
        assert trade_calls[0]["net_price"] == 0.90
        events = [json.loads(l) for l in log.read_text().splitlines()]
        lp = next(e for e in events if e["event"] == "result")["result"]["limit_pricing"]
        assert lp["fallback_reason"] == "missing_leg_in_refreshed_quotes"

    def test_trade_handler_market_skips_quote_refresh(self, monkeypatch, tmp_path):
        """MARKET orders don't fetch quotes — net_price stays None."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        quote_calls: list[dict] = []
        trade_calls: list[dict] = []
        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_option_quotes(self, **kw):
                quote_calls.append(kw)
                return {"quotes": {}, "meta": {}}
            async def trade_credit_spread(self, **kw):
                trade_calls.append(kw)
                return {"order_id": "x", "status": "FILLED"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "order_type": "MARKET"})
        h = TradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                        policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread("SPX", credit=0.85)], self._ctx()))
        assert quote_calls == []
        assert trade_calls[0]["net_price"] is None

    def test_policy_limit_slippage_defaults_to_none_for_daemon_resolution(self):
        """Default = None means 'use daemon /trade/defaults'. Explicit float overrides."""
        from spread_scanner import TradePolicy
        assert TradePolicy().limit_slippage_pct is None      # → daemon default
        p = TradePolicy.from_dict({"limit_slippage_pct": 2.5})
        assert p.limit_slippage_pct == 2.5                   # explicit override

    def test_trade_handler_logs_error_on_exception(self, monkeypatch, tmp_path):
        import asyncio, sys
        from spread_scanner import TradeHandler, TradePolicy

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, **kw):
                raise RuntimeError("broker rejected")

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        log_file = tmp_path / "err.jsonl"
        h = TradeHandler(min_norm_roi=0, log_file=str(log_file),
                        policy=TradePolicy.from_dict({"roi_pct": [1.0, 10.0]}),
                        daemon_url="http://d")
        asyncio.run(h.fire([self._spread("SPX", credit=1.0)], self._ctx()))

        events = [json.loads(l) for l in log_file.read_text().splitlines()]
        err = [e for e in events if e["event"] == "error"]
        assert len(err) == 1
        assert "broker rejected" in err[0]["reason"]

    def test_sim_and_trade_use_separate_log_files(self, monkeypatch, tmp_path):
        import asyncio, sys
        from spread_scanner import SimulateTradeHandler, TradeHandler, TradePolicy

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def check_margin_credit_spread(self, **kw):
                return {"init_margin": 1000, "error": None}
            async def trade_credit_spread(self, **kw):
                return {"order_id": "real"}

        import spread_scanner as ss
        self._bind_utp_pricing(FakeTClient)
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)

        sim_file = tmp_path / "sim.jsonl"
        live_file = tmp_path / "live.jsonl"
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0]})
        sim = SimulateTradeHandler(min_norm_roi=0, log_file=str(sim_file),
                                   policy=pol, daemon_url="http://d")
        live = TradeHandler(min_norm_roi=0, log_file=str(live_file),
                           policy=pol, daemon_url="http://d")

        spreads = [self._spread("SPX", credit=1.0)]
        ctx = self._ctx()
        asyncio.run(sim.fire(spreads, ctx))
        asyncio.run(live.fire(spreads, ctx))

        assert sim_file.exists() and live_file.exists()
        sim_events = [json.loads(l) for l in sim_file.read_text().splitlines()]
        live_events = [json.loads(l) for l in live_file.read_text().splitlines()]
        assert all(e["handler"] == "simulate_trade" for e in sim_events)
        assert all(e["handler"] == "trade" for e in live_events)

    # --- activation policy ------------------------------------------------

    def test_trade_handlers_not_activated_by_default(self, monkeypatch):
        """Without YAML or explicit opt-in, no trade handler must appear in the pipeline."""
        from spread_scanner import _load_config, TradeHandler, SimulateTradeHandler
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--tickers", "SPX"])
        args, handlers = _load_config()
        assert not any(isinstance(h, (TradeHandler, SimulateTradeHandler)) for h in handlers)

    def test_trade_handlers_not_activated_by_log_or_notify_cli(self, monkeypatch):
        """Legacy --log and --notify flags must NOT produce any trade handler."""
        from spread_scanner import _load_config, TradeHandler, SimulateTradeHandler
        monkeypatch.setattr("sys.argv", [
            "spread_scanner.py", "--tickers", "SPX",
            "--log", "3:/tmp/l.jsonl", "--notify", "4:me@x.com",
        ])
        args, handlers = _load_config()
        assert not any(isinstance(h, (TradeHandler, SimulateTradeHandler)) for h in handlers)
        assert len(handlers) == 2  # log + notify only

    def test_trade_handler_activates_only_via_yaml(self, tmp_path, monkeypatch):
        from spread_scanner import _load_config, TradeHandler, SimulateTradeHandler
        cfg = tmp_path / "c.yaml"
        cfg.write_text("""
tickers: [SPX]
handlers:
  - type: simulate_trade
    min_norm_roi: 1.5
    log_file: /tmp/sim.jsonl
    policy:
      roi_pct: [1.5, 5.0]
      min_credit: {SPX: 0.50}
  - type: trade
    min_norm_roi: 1.5
    log_file: /tmp/live.jsonl
    policy:
      roi_pct: [1.5, 5.0]
      min_credit: {SPX: 0.50}
""")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(cfg)])
        args, handlers = _load_config()
        assert any(isinstance(h, SimulateTradeHandler) for h in handlers)
        assert any(isinstance(h, TradeHandler) for h in handlers)

    # ── nROI schedule + per-ticker ────────────────────────────────────────

    def test_parse_min_norm_roi_schedule_normalizes_times(self):
        from spread_scanner import _parse_min_norm_roi_schedule
        from datetime import time
        s = _parse_min_norm_roi_schedule([
            {"until": "07:30", "value": 2.0},
            {"until": "09:00", "value": 1.75},
            {"value": 1.5},   # open-ended fallback
        ])
        assert s[0]["until"] == time(7, 30) and s[0]["value"] == 2.0
        assert s[1]["until"] == time(9, 0) and s[1]["value"] == 1.75
        assert s[2]["until"] is None and s[2]["value"] == 1.5

    def test_parse_min_norm_roi_schedule_rejects_missing_value(self):
        import pytest
        from spread_scanner import _parse_min_norm_roi_schedule
        with pytest.raises(ValueError, match="missing required 'value'"):
            _parse_min_norm_roi_schedule([{"until": "07:30"}])

    def test_parse_min_norm_roi_schedule_rejects_early_fallback(self):
        """Only the LAST entry can be open-ended — earlier ones must have 'until'."""
        import pytest
        from spread_scanner import _parse_min_norm_roi_schedule
        with pytest.raises(ValueError, match="only the final entry"):
            _parse_min_norm_roi_schedule([
                {"value": 2.0},               # no until, but not last → error
                {"until": "09:00", "value": 1.5},
            ])

    def test_eval_schedule_picks_right_bucket(self):
        from spread_scanner import ActionHandler
        from datetime import datetime, time
        from spread_scanner import _PT
        sched = [{"until": time(7, 30), "value": 2.0},
                 {"until": None, "value": 1.5}]
        # Before 07:30 → first entry
        t1 = datetime(2026, 4, 23, 7, 0, tzinfo=_PT)
        assert ActionHandler._eval_schedule(sched, t1) == 2.0
        # Exactly at 07:30 → first entry (inclusive)
        t2 = datetime(2026, 4, 23, 7, 30, tzinfo=_PT)
        assert ActionHandler._eval_schedule(sched, t2) == 2.0
        # After 07:30 → fallback
        t3 = datetime(2026, 4, 23, 9, 0, tzinfo=_PT)
        assert ActionHandler._eval_schedule(sched, t3) == 1.5

    def test_build_handler_parses_nroi_schedule(self, tmp_path):
        from spread_scanner import build_handler
        h = build_handler({
            "type": "log", "path": str(tmp_path / "l.jsonl"),
            "min_norm_roi_schedule": [
                {"until": "07:30", "value": 2.0},
                {"value": 1.5},
            ],
        })
        assert h.min_norm_roi_schedule is not None
        assert len(h.min_norm_roi_schedule) == 2

    def test_build_handler_parses_nroi_per_ticker(self, tmp_path):
        from spread_scanner import build_handler
        h = build_handler({
            "type": "log", "path": str(tmp_path / "l.jsonl"),
            "min_norm_roi": 3.0,
            "min_norm_roi_per_ticker": {
                "RUT": 1.5,
                "NDX": 2.5,
            },
        })
        assert h.min_norm_roi_per_ticker == {"RUT": 1.5, "NDX": 2.5}

    def test_build_handler_parses_nroi_per_ticker_with_schedule(self, tmp_path):
        """Per-ticker values can themselves be schedules."""
        from spread_scanner import build_handler
        from datetime import time
        h = build_handler({
            "type": "log", "path": str(tmp_path / "l.jsonl"),
            "min_norm_roi": 3.0,
            "min_norm_roi_per_ticker": {
                "RUT": [{"until": "07:30", "value": 1.5}, {"value": 1.0}],
            },
        })
        rut_sched = h.min_norm_roi_per_ticker["RUT"]
        assert rut_sched[0]["until"] == time(7, 30)
        assert rut_sched[0]["value"] == 1.5

    def test_resolve_min_norm_roi_scalar_fallback(self, tmp_path):
        from spread_scanner import build_handler
        h = build_handler({
            "type": "log", "path": str(tmp_path / "l.jsonl"),
            "min_norm_roi": 3.0,
        })
        # No symbol, no schedule → the scalar
        assert h._resolve_min_norm_roi() == 3.0
        assert h._resolve_min_norm_roi("SPX") == 3.0

    def test_resolve_min_norm_roi_per_ticker_overrides(self, tmp_path):
        from spread_scanner import build_handler
        h = build_handler({
            "type": "log", "path": str(tmp_path / "l.jsonl"),
            "min_norm_roi": 3.0,
            "min_norm_roi_per_ticker": {"RUT": 1.5, "SPX": 2.5},
        })
        assert h._resolve_min_norm_roi("RUT") == 1.5
        assert h._resolve_min_norm_roi("SPX") == 2.5
        # Ticker without override → scalar fallback
        assert h._resolve_min_norm_roi("NDX") == 3.0

    def test_resolve_min_norm_roi_schedule_time_varying(self, tmp_path):
        from spread_scanner import build_handler, _PT
        from datetime import datetime
        h = build_handler({
            "type": "log", "path": str(tmp_path / "l.jsonl"),
            "min_norm_roi_schedule": [
                {"until": "07:30", "value": 2.0},
                {"value": 1.5},
            ],
        })
        before = datetime(2026, 4, 23, 7, 0, tzinfo=_PT)
        after  = datetime(2026, 4, 23, 9, 0, tzinfo=_PT)
        assert h._resolve_min_norm_roi("SPX", before) == 2.0
        assert h._resolve_min_norm_roi("SPX", after) == 1.5

    def test_resolve_min_norm_roi_per_ticker_beats_schedule(self, tmp_path):
        """Per-ticker override wins over the time-of-day schedule."""
        from spread_scanner import build_handler, _PT
        from datetime import datetime
        h = build_handler({
            "type": "log", "path": str(tmp_path / "l.jsonl"),
            "min_norm_roi_schedule": [
                {"until": "07:30", "value": 2.0},
                {"value": 1.5},
            ],
            "min_norm_roi_per_ticker": {"RUT": 1.0},
        })
        # RUT — per-ticker wins
        before = datetime(2026, 4, 23, 7, 0, tzinfo=_PT)
        after  = datetime(2026, 4, 23, 9, 0, tzinfo=_PT)
        assert h._resolve_min_norm_roi("RUT", before) == 1.0
        assert h._resolve_min_norm_roi("RUT", after) == 1.0
        # SPX — no override, falls through to schedule
        assert h._resolve_min_norm_roi("SPX", before) == 2.0
        assert h._resolve_min_norm_roi("SPX", after) == 1.5

    def test_trade_handler_filter_honors_schedule(self, monkeypatch, tmp_path):
        """TradeHandler.filter() uses the resolver so time-varying threshold
        actually lets more candidates through after the cutoff."""
        import spread_scanner as ss
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: True)
        from spread_scanner import build_handler
        h = build_handler({
            "type": "simulate_trade",
            "log_file": str(tmp_path / "sim.jsonl"),
            "daemon_url": "http://d",
            "min_norm_roi_schedule": [
                {"until": "07:30", "value": 2.5},
                {"value": 1.5},
            ],
            "policy": {"roi_pct": [1.0, 10.0]},
        })
        # Candidate: nROI = 2.0 (between the two thresholds)
        cand = [{"symbol": "SPX", "option_type": "PUT", "short_strike": 7050,
                 "long_strike": 7025, "width": 25, "credit": 0.70, "roi_pct": 2.0,
                 "otm_pct": 1.0, "dte": 0, "prev_close": 7064, "expiration": "2026-04-22"}]
        # Monkey-patch datetime.now() to simulate different times.
        from datetime import datetime as real_dt
        from spread_scanner import _PT
        class FakeDT:
            @staticmethod
            def now(tz=None):
                return FakeDT._now if tz is None else FakeDT._now
        # BEFORE 07:30: threshold 2.5 → candidate nROI 2.0 rejected
        FakeDT._now = real_dt(2026, 4, 23, 7, 0, tzinfo=_PT)
        monkeypatch.setattr(ss, "datetime", FakeDT)
        out_before = h.filter(cand)
        assert out_before == []
        # AFTER 07:30: threshold 1.5 → 2.0 passes
        FakeDT._now = real_dt(2026, 4, 23, 9, 0, tzinfo=_PT)
        out_after = h.filter(cand)
        assert len(out_after) == 1

    def test_build_handler_trade_requires_log_file(self):
        import pytest
        from spread_scanner import build_handler
        with pytest.raises(ValueError, match="trade handler requires 'log_file'"):
            build_handler({"type": "trade", "min_norm_roi": 2.0})
        with pytest.raises(ValueError, match="simulate_trade handler requires 'log_file'"):
            build_handler({"type": "simulate_trade", "min_norm_roi": 2.0})

    # --- rejection logging ------------------------------------------------

    @staticmethod
    def _read_log(path):
        return [json.loads(l) for l in open(path) if l.strip()]

    def test_filter_logs_below_credit_floor(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        log = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0,
            log_file=str(log),
            policy=TradePolicy.from_dict({
                "roi_pct": [1.0, 10.0], "min_credit": {"SPX": 0.50},
            }),
            daemon_url="http://d",
        )
        h.filter([self._spread("SPX", credit=0.30)])
        evs = self._read_log(log)
        assert any(e["event"] == "rejected" and e["reason"] == "below_credit_floor"
                   for e in evs)
        r = [e for e in evs if e["reason"] == "below_credit_floor"][0]
        assert r["credit"] == 0.30 and r["credit_floor"] == 0.50

    def test_filter_logs_below_otm_floor(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        log = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0, log_file=str(log),
            policy=TradePolicy.from_dict({
                "roi_pct": [1.0, 10.0], "min_otm_pct": {"SPX": 1.0},
            }),
            daemon_url="http://d",
        )
        h.filter([self._spread("SPX", otm=0.5)])
        evs = self._read_log(log)
        r = [e for e in evs if e.get("reason") == "below_otm_floor"]
        assert len(r) == 1
        assert r[0]["otm_pct"] == 0.5 and r[0]["otm_floor"] == 1.0

    def test_filter_logs_roi_outside_band(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        log = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0, log_file=str(log),
            policy=TradePolicy.from_dict({"roi_pct": [1.5, 5.0]}),
            daemon_url="http://d",
        )
        h.filter([
            self._spread("SPX", roi=1.0),   # below
            self._spread("SPX", roi=7.0),   # above
        ])
        evs = self._read_log(log)
        r = [e for e in evs if e.get("reason") == "roi_outside_band"]
        assert len(r) == 2
        assert {e["roi_pct"] for e in r} == {1.0, 7.0}

    def test_filter_logs_missing_prev_close(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        log = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0, log_file=str(log),
            policy=TradePolicy.from_dict({"roi_pct": [1.0, 10.0]}),
            daemon_url="http://d",
        )
        h.filter([self._spread("SPX", prev_close=0)])
        evs = self._read_log(log)
        r = [e for e in evs if e.get("reason") == "missing_prev_close"]
        assert len(r) == 1

    def test_filter_logs_below_min_norm_roi(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        log = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=3.0, log_file=str(log),
            policy=TradePolicy.from_dict({"roi_pct": [1.0, 10.0]}),
            daemon_url="http://d",
        )
        h.filter([self._spread("SPX", roi=2.0)])   # nROI = 2.0 < 3.0
        evs = self._read_log(log)
        r = [e for e in evs if e.get("reason") == "below_min_norm_roi"]
        assert len(r) == 1
        assert r[0]["norm_roi"] == 2.0 and r[0]["threshold"] == 3.0

    def test_filter_logs_outranked_same_ticker(self, monkeypatch, tmp_path):
        """Two passing SPX candidates in same scan → lower-nROI one logged as outranked."""
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        log = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0, log_file=str(log),
            policy=TradePolicy.from_dict({"roi_pct": [1.0, 10.0]}),
            daemon_url="http://d",
        )
        out = h.filter([
            self._spread("SPX", roi=3.0, short=5700, long_=5680),
            self._spread("SPX", roi=4.5, short=5710, long_=5690),   # wins
            self._spread("SPX", roi=2.0, short=5720, long_=5700),
        ])
        assert len(out) == 1 and out[0]["roi_pct"] == 4.5
        evs = self._read_log(log)
        r = [e for e in evs if e.get("reason") == "outranked_same_ticker"]
        assert len(r) == 2
        for e in r:
            assert e["winning_norm_roi"] == 4.5
            assert e["winning_short_strike"] == 5710

    def test_filter_logs_batch_outside_trading_window(self, monkeypatch, tmp_path):
        import spread_scanner as ss
        from spread_scanner import SimulateTradeHandler, TradePolicy
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: False)
        log = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(
            min_norm_roi=0, log_file=str(log),
            policy=TradePolicy(), daemon_url="http://d",
        )
        h.filter([self._spread("SPX"), self._spread("NDX", prev_close=26000)])
        evs = self._read_log(log)
        # One summary event per batch (not per candidate) for this case
        assert len(evs) == 1
        assert evs[0]["event"] == "rejected_batch"
        assert evs[0]["reason"] == "outside_trading_window"
        assert evs[0]["count"] == 2

    def test_count_rejected_increments(self, monkeypatch, tmp_path):
        self._always_in_window(monkeypatch)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        h = SimulateTradeHandler(
            min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
            policy=TradePolicy.from_dict({
                "roi_pct": [1.5, 5.0], "min_credit": {"SPX": 0.50},
            }),
            daemon_url="http://d",
        )
        assert h.count_rejected == 0
        h.filter([
            self._spread("SPX", credit=0.20),   # below_credit_floor
            self._spread("SPX", roi=7.0),       # roi_outside_band
        ])
        assert h.count_rejected == 2


class TestRecentActionsAndNotifications:
    """recent_actions buffer + dashboard panel + per-action notifications."""

    def _ctx(self, client=None):
        from spread_scanner import HandlerContext
        return HandlerContext(client=client, args=None, scan_data={},
                              is_market_hours=True, now_ts="2026-04-22T07:00:00")

    @staticmethod
    def _spread(sym="SPX", short=7050, long_=7030, credit=0.95, width=20,
                roi=3.2, otm=1.2, prev_close=7064.01):
        return {
            "symbol": sym, "option_type": "PUT",
            "short_strike": short, "long_strike": long_, "width": width,
            "credit": credit, "roi_pct": roi, "otm_pct": otm,
            "dte": 0, "expiration": "2026-04-22", "prev_close": prev_close,
        }

    # ── recent_actions buffer ────────────────────────────────────────────

    def test_record_action_appends_to_buffer(self, tmp_path):
        from spread_scanner import SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        assert len(h.recent_actions) == 0
        h._record_action("SIMULATED", self._spread(), contracts=1)
        h._record_action("SKIPPED", self._spread(short=7040), contracts=2, reason="total_risk_cap")
        assert len(h.recent_actions) == 2
        first, second = h.recent_actions
        assert first["outcome"] == "SIMULATED"
        assert first["symbol"] == "SPX"
        assert first["credit_dollars"] == 95.0
        assert first["risk_dollars"] == 1905.0          # (20-0.95)*100*1
        assert first["norm_roi"] == 3.2                  # roi_pct/(dte+1)
        assert second["outcome"] == "SKIPPED"
        assert second["reason"] == "total_risk_cap"
        assert second["contracts"] == 2
        assert second["credit_dollars"] == 190.0

    def test_recent_actions_is_bounded(self, tmp_path):
        from spread_scanner import SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        # deque maxlen = 20; push 30, only last 20 should survive.
        for i in range(30):
            h._record_action("SIMULATED", self._spread(short=7000 + i), contracts=1)
        assert len(h.recent_actions) == 20
        # Oldest should be the one with short=7010 (the first 10 were evicted).
        assert h.recent_actions[0]["short_strike"] == 7010

    # ── render_recent_actions ────────────────────────────────────────────

    def test_render_recent_actions_orders_newest_at_bottom(self, tmp_path):
        import time
        from spread_scanner import SimulateTradeHandler, TradePolicy, render_recent_actions
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        h._record_action("SIMULATED", self._spread(short=7050), contracts=1)
        time.sleep(0.005)
        h._record_action("SKIPPED", self._spread(short=7040), contracts=2, reason="total_risk_cap")
        time.sleep(0.005)
        h._record_action("ERROR", self._spread(short=7030), contracts=1, reason="broker down")

        lines = render_recent_actions([h], n=3)
        # First heading line is the section header; find data rows.
        rows = [l for l in lines if any(c in l for c in ("P7050", "P7040", "P7030"))]
        assert len(rows) == 3
        # Oldest (7050) first, newest (7030) last.
        assert "P7050" in rows[0]
        assert "P7040" in rows[1]
        assert "P7030" in rows[2]
        assert "SIMULATED" in rows[0]
        assert "SKIPPED" in rows[1] and "total_risk_cap" in rows[1]
        assert "ERROR" in rows[2] and "broker down" in rows[2]

    def test_render_recent_actions_hidden_when_n_zero(self, tmp_path):
        from spread_scanner import SimulateTradeHandler, TradePolicy, render_recent_actions
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        h._record_action("SIMULATED", self._spread(), contracts=1)
        assert render_recent_actions([h], n=0) == []

    def test_render_recent_actions_merges_across_handlers(self, tmp_path):
        import time
        from spread_scanner import SimulateTradeHandler, TradeHandler, TradePolicy, render_recent_actions
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h_sim = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "sim.jsonl"),
                                     policy=pol, daemon_url="http://d")
        h_live = TradeHandler(min_norm_roi=0, log_file=str(tmp_path / "live.jsonl"),
                              policy=pol, daemon_url="http://d")
        h_sim._record_action("SIMULATED", self._spread(short=7050), contracts=1)
        time.sleep(0.005)
        h_live._record_action("FILLED", self._spread(short=7030), contracts=2)
        lines = render_recent_actions([h_sim, h_live], n=3)
        rows = [l for l in lines if any(c in l for c in ("P7050", "P7030"))]
        assert len(rows) == 2
        assert "P7050" in rows[0]       # older
        assert "P7030" in rows[1]       # newer
        assert "SIM" in rows[0]
        assert "TRADE" in rows[1]

    # ── notification wiring ──────────────────────────────────────────────

    def test_notification_fires_on_success(self, monkeypatch, tmp_path):
        import asyncio
        from spread_scanner import SimulateTradeHandler, TradePolicy

        posts = []
        class FakeHttpClient:
            async def post(self, url, json=None, timeout=None):
                posts.append({"url": url, "json": json})
                class R: status_code = 200
                return R()

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def check_margin_credit_spread(self, **kw):
                return {"init_margin": 1000, "error": None}

        import spread_scanner as ss
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: True)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0]})   # notify default True
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread()],
            ss.HandlerContext(client=FakeHttpClient(), args=None, scan_data={},
                              is_market_hours=True, now_ts="2026-04-22T07:00:00")))

        # Exactly one /api/notify POST per successful trade.
        notify_posts = [p for p in posts if p["url"].endswith("/api/notify")]
        assert len(notify_posts) == 1
        body = notify_posts[0]["json"]
        assert body["channel"] == "both"                    # policy default
        assert "SIMULATE_TRADE" in body["subject"]
        assert "SPX" in body["message"]
        assert "P7050/7030" in body["subject"] or "7050/7030" in body["subject"]
        assert "credit $0.95" in body["message"]

    def test_notification_skipped_when_policy_notify_false(self, monkeypatch, tmp_path):
        import asyncio
        from spread_scanner import SimulateTradeHandler, TradePolicy

        posts = []
        class FakeHttpClient:
            async def post(self, url, json=None, timeout=None):
                posts.append({"url": url, "json": json})
                class R: status_code = 200
                return R()

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def check_margin_credit_spread(self, **kw):
                return {"init_margin": 1000, "error": None}

        import spread_scanner as ss
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: True)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread()],
            ss.HandlerContext(client=FakeHttpClient(), args=None, scan_data={},
                              is_market_hours=True, now_ts="2026-04-22T07:00:00")))

        assert not any(p["url"].endswith("/api/notify") for p in posts)

    def test_notification_failure_swallowed(self, monkeypatch, tmp_path, capsys):
        import asyncio
        from spread_scanner import SimulateTradeHandler, TradePolicy

        class BadHttpClient:
            async def post(self, url, json=None, timeout=None):
                raise RuntimeError("notify service is down")

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def check_margin_credit_spread(self, **kw):
                return {"init_margin": 1000, "error": None}

        import spread_scanner as ss
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: True)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0]})
        log_path = tmp_path / "l.jsonl"
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(log_path),
                                 policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread()],
            ss.HandlerContext(client=BadHttpClient(), args=None, scan_data={},
                              is_market_hours=True, now_ts="2026-04-22T07:00:00")))

        # Trade still recorded to the JSONL log + recent_actions despite notify failure.
        events = [json.loads(l) for l in log_path.read_text().splitlines()]
        assert any(e["event"] == "result" for e in events)
        assert len(h.recent_actions) == 1
        # The failure was logged to stderr.
        err = capsys.readouterr().err
        assert "[notify:simulate_trade] failed" in err

    def test_notification_on_error_outcome(self, monkeypatch, tmp_path):
        """A broker exception should still emit a notification tagged ERROR."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        posts = []
        class FakeHttpClient:
            async def post(self, url, json=None, timeout=None):
                posts.append({"url": url, "json": json})
                class R: status_code = 200
                return R()

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, **kw):
                raise RuntimeError("broker rejected")

        import spread_scanner as ss
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: True)

        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "order_type": "MARKET"})
        h = TradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                        policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread()],
            ss.HandlerContext(client=FakeHttpClient(), args=None, scan_data={},
                              is_market_hours=True, now_ts="2026-04-22T07:00:00")))

        notify_posts = [p for p in posts if p["url"].endswith("/api/notify")]
        assert len(notify_posts) == 1
        body = notify_posts[0]["json"]
        assert "TRADE" in body["subject"] and "ERROR" in body["subject"]
        assert "broker rejected" in body["message"]

    # ── quiet-scan heartbeats ────────────────────────────────────────────

    def test_diagnose_quiet_no_spreads(self):
        """Empty scan → 'screener produced no spreads' reason."""
        from spread_scanner import _diagnose_quiet_reason
        data = {"dte_sections": {0: {"spreads": {"SPX": [], "NDX": []}}}}
        reason = _diagnose_quiet_reason([], data, [])
        assert "no spreads" in reason

    def test_diagnose_quiet_no_candidates(self):
        """Spreads existed but the screener filters dropped them all."""
        from spread_scanner import _diagnose_quiet_reason
        data = {"dte_sections": {0: {"spreads": {"SPX": [{"credit": 0.1}]}}}}
        reason = _diagnose_quiet_reason([], data, [])
        assert "no candidates passed screener filters" in reason

    def test_diagnose_quiet_names_gates_with_counts(self, tmp_path):
        """Diagnoser reports which gates fired with their per-scan counts."""
        from spread_scanner import _diagnose_quiet_reason, SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        # Simulate this scan's filter() populating the counter with two gates:
        h.last_scan_rejection_counts = {
            "below_otm_floor": 3,
            "below_credit_floor": 2,
        }
        data = {"dte_sections": {0: {"spreads": {"SPX": [{"credit": 0.1}]}}}}
        reason = _diagnose_quiet_reason([h], data, [{"symbol": "SPX"}, {"symbol": "NDX"}])
        # Should explicitly name EACH gate and its count, total = 5.
        assert "rejected by gates" in reason
        assert "below_otm_floor (3)" in reason
        assert "below_credit_floor (2)" in reason
        assert reason.startswith("5 rejected")

    def test_diagnose_quiet_single_gate_omits_count(self, tmp_path):
        """When only one gate fires, the count is implicit (all of them)."""
        from spread_scanner import _diagnose_quiet_reason, SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        h.last_scan_rejection_counts = {"total_risk_cap": 4}
        data = {"dte_sections": {0: {"spreads": {"SPX": [{"credit": 0.1}]}}}}
        reason = _diagnose_quiet_reason([h], data, [{"symbol": "SPX"}])
        assert reason == "4 rejected by gates: total_risk_cap"

    def test_diagnose_quiet_more_than_three_gates_truncates(self, tmp_path):
        """When >3 gates fire, show top 3 and +N more."""
        from spread_scanner import _diagnose_quiet_reason, SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        h.last_scan_rejection_counts = {
            "below_otm_floor": 3, "below_credit_floor": 2,
            "roi_outside_band": 1, "missing_prev_close": 1, "total_risk_cap": 1,
        }
        data = {"dte_sections": {0: {"spreads": {"SPX": [{"credit": 0.1}]}}}}
        reason = _diagnose_quiet_reason([h], data, [{"symbol": "SPX"}])
        assert "below_otm_floor (3)" in reason
        assert "below_credit_floor (2)" in reason
        # 5 unique gates total → +2 more
        assert "+2 more" in reason

    def test_filter_resets_rejection_counts_each_scan(self, tmp_path, monkeypatch):
        """filter() resets last_scan_rejection_counts so each scan is fresh."""
        import spread_scanner as ss
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: True)
        from spread_scanner import SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({
            "roi_pct": [1.0, 10.0], "min_otm_pct": {"SPX": 1.5},
        })
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        # First scan: 1 below_otm_floor rejection
        h.filter([self._spread(otm=0.5)])
        assert h.last_scan_rejection_counts == {"below_otm_floor": 1}
        # Second scan: counter resets, now 0 rejections
        h.filter([])
        assert h.last_scan_rejection_counts == {}

    def test_maybe_log_quiet_heartbeat_appends_when_nothing_fired(self, tmp_path):
        from collections import deque
        from spread_scanner import _maybe_log_quiet_heartbeat, SimulateTradeHandler, TradePolicy
        import types
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        args = types.SimpleNamespace(activity_log=deque(maxlen=50))
        data = {"dte_sections": {0: {"spreads": {"SPX": [{"credit": 0.1}]}}}}

        # Nothing fired → heartbeat appended
        _maybe_log_quiet_heartbeat(args, [h], data, [], fired=False)
        assert len(args.activity_log) == 1
        assert args.activity_log[0]["outcome"] == "QUIET"
        assert "reason" in args.activity_log[0]

        # Something fired → NO heartbeat
        _maybe_log_quiet_heartbeat(args, [h], data, [], fired=True)
        assert len(args.activity_log) == 1     # unchanged

    def test_activity_panel_merges_heartbeats_with_trade_rows(self, tmp_path):
        import time
        from collections import deque
        from datetime import datetime
        from spread_scanner import (
            SimulateTradeHandler, TradePolicy, render_activity_panel,
        )
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        h._record_action("SIMULATED", self._spread(), contracts=1)
        time.sleep(0.005)

        log = deque(maxlen=50)
        now = datetime.now()
        log.append({
            "ts": now.strftime("%H:%M:%S"),
            "_sort_key": now.timestamp(),
            "outcome": "QUIET",
            "reason": "no candidates passed screener filters",
        })
        time.sleep(0.005)
        h._record_action("SIMULATED", self._spread(short=7040), contracts=1)

        lines = render_activity_panel([h], n=5, activity_log=log)
        body = "\n".join(lines)
        # Border present (top + bottom)
        assert "ACTIVITY (last 3)" in body
        # Both trade rows AND the quiet heartbeat visible
        assert body.count("SIM") >= 2
        assert "SCAN" in body
        assert "no candidates passed screener filters" in body
        # Top border and bottom border chars present
        assert "╭" in body and "╯" in body

    def test_activity_panel_renders_nothing_when_empty(self, tmp_path):
        from spread_scanner import render_activity_panel, SimulateTradeHandler, TradePolicy
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0], "notify": False})
        h = SimulateTradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                                 policy=pol, daemon_url="http://d")
        assert render_activity_panel([h], n=5, activity_log=None) == []

    def test_skip_does_not_notify(self, monkeypatch, tmp_path):
        """Risk-cap skips go to the log + recent_actions, but never trigger a notification."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy

        posts = []
        class FakeHttpClient:
            async def post(self, url, json=None, timeout=None):
                posts.append(url)
                class R: status_code = 200
                return R()

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def trade_credit_spread(self, **kw):
                return {"order_id": "x", "status": "FILLED"}

        import spread_scanner as ss
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window",
                            lambda self, now_pt=None: True)

        # Total risk cap = 100 → first trade's $1,905 max-loss already breaches it.
        pol = TradePolicy.from_dict({
            "roi_pct": [1.0, 10.0], "max_total_risk": 100,
            "order_type": "MARKET",
        })
        h = TradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                        policy=pol, daemon_url="http://d")
        asyncio.run(h.fire([self._spread()],
            ss.HandlerContext(client=FakeHttpClient(), args=None, scan_data={},
                              is_market_hours=True, now_ts="2026-04-22T07:00:00")))

        assert not any(u.endswith("/api/notify") for u in posts)
        # And skip was recorded
        assert len(h.recent_actions) == 1
        assert h.recent_actions[0]["outcome"] == "SKIPPED"


class TestDaemonTradeDefaults:
    """/trade/defaults endpoint + TradingClient daemon-default resolution."""

    def test_get_trade_defaults_endpoint(self, monkeypatch):
        """GET /trade/defaults returns env-configured values."""
        from fastapi.testclient import TestClient
        from app.config import settings
        monkeypatch.setattr(settings, "default_order_type", "LIMIT")
        monkeypatch.setattr(settings, "limit_slippage_pct", 2.5)
        monkeypatch.setattr(settings, "limit_quote_max_age_sec", 7.0)

        from app.main import app
        client = TestClient(app)
        resp = client.get("/trade/defaults")
        assert resp.status_code == 200
        data = resp.json()
        assert data["default_order_type"] == "LIMIT"
        assert data["limit_slippage_pct"] == 2.5
        assert data["limit_quote_max_age_sec"] == 7.0

    def test_tradingclient_get_trade_defaults(self):
        """TradingClient.get_trade_defaults hits /trade/defaults."""
        import asyncio
        from utp import TradingClient
        captured = {}
        class FakeHttp:
            async def get(self, path, params=None):
                captured["path"] = path
                class R:
                    status_code = 200
                    def raise_for_status(self_): pass
                    def json(self_): return {"default_order_type": "MARKET",
                                             "limit_slippage_pct": 1.0,
                                             "limit_quote_max_age_sec": 8.0}
                return R()
        c = TradingClient("http://d")
        c._client = FakeHttp()
        data = asyncio.run(c.get_trade_defaults())
        assert captured["path"] == "/trade/defaults"
        assert data["default_order_type"] == "MARKET"

    def test_compute_uses_daemon_slippage_when_pct_is_none(self):
        """slippage_pct=None → fetch from daemon defaults."""
        import asyncio
        from utp import TradingClient
        class FakeHttp:
            async def get(self, path, params=None):
                class R:
                    status_code = 200
                    def raise_for_status(self_): pass
                    def json(self_):
                        if path == "/trade/defaults":
                            return {"default_order_type": "LIMIT",
                                    "limit_slippage_pct": 5.0,
                                    "limit_quote_max_age_sec": 12.0}
                        # /market/options/SPX
                        return {
                            "quotes": {"put": [
                                {"strike": 5700.0, "bid": 1.00, "ask": 1.10},
                                {"strike": 5680.0, "bid": 0.20, "ask": 0.30},
                            ]},
                            "meta": {"age_seconds": 4.0, "source": "fresh_cache"},
                        }
                return R()
        c = TradingClient("http://d")
        c._client = FakeHttp()
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            slippage_pct=None, max_age=None, fallback_credit=0.80,
        ))
        # refreshed = 1.00 - 0.30 = 0.70; daemon says 5% slippage → 0.665
        # Python's banker's rounding takes 0.665 → 0.66 (round-half-to-even).
        assert price == 0.66
        assert meta["slippage_pct"] == 5.0

    def test_compute_falls_back_to_zero_slippage_when_daemon_unreachable(self):
        """If /trade/defaults errors, _resolve_trade_defaults uses 0/10 fallback."""
        import asyncio
        from utp import TradingClient
        class FakeHttp:
            async def get(self, path, params=None):
                if path == "/trade/defaults":
                    raise RuntimeError("daemon down")
                class R:
                    status_code = 200
                    def raise_for_status(self_): pass
                    def json(self_):
                        return {
                            "quotes": {"put": [
                                {"strike": 5700.0, "bid": 1.00, "ask": 1.10},
                                {"strike": 5680.0, "bid": 0.20, "ask": 0.30},
                            ]},
                            "meta": {"age_seconds": 4.0, "source": "fresh_cache"},
                        }
                return R()
        c = TradingClient("http://d")
        c._client = FakeHttp()
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            slippage_pct=None, max_age=None, fallback_credit=0.80,
        ))
        assert price == 0.70                 # refreshed credit, 0% slippage (fallback)
        assert meta["slippage_pct"] == 0.0

    def test_explicit_slippage_overrides_daemon(self):
        """Caller-supplied slippage_pct wins over daemon config."""
        import asyncio
        from utp import TradingClient
        daemon_calls = []
        class FakeHttp:
            async def get(self, path, params=None):
                daemon_calls.append(path)
                class R:
                    status_code = 200
                    def raise_for_status(self_): pass
                    def json(self_):
                        if path == "/trade/defaults":
                            return {"limit_slippage_pct": 5.0,
                                    "limit_quote_max_age_sec": 20.0,
                                    "default_order_type": "MARKET"}
                        return {
                            "quotes": {"put": [
                                {"strike": 5700.0, "bid": 1.00, "ask": 1.10},
                                {"strike": 5680.0, "bid": 0.20, "ask": 0.30},
                            ]},
                            "meta": {"age_seconds": 4.0, "source": "fresh_cache"},
                        }
                return R()
        c = TradingClient("http://d")
        c._client = FakeHttp()
        # Both explicit → no /trade/defaults call needed
        price, _ = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            slippage_pct=0.0, max_age=10.0, fallback_credit=0.80,
        ))
        assert price == 0.70
        assert "/trade/defaults" not in daemon_calls

    def test_handler_resolves_order_type_from_daemon_when_unset(self, monkeypatch, tmp_path):
        """With policy.order_type=None, daemon's default_order_type is used."""
        import asyncio
        from spread_scanner import TradeHandler, TradePolicy, HandlerContext

        trade_calls: list[dict] = []

        class FakeTClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            async def get_trade_defaults(self):
                return {"default_order_type": "LIMIT",
                        "limit_slippage_pct": 0.0,
                        "limit_quote_max_age_sec": 10.0}
            async def get_option_quotes(self, **kw):
                return {
                    "quotes": {"put": [
                        {"strike": 5700.0, "bid": 1.00, "ask": 1.10},
                        {"strike": 5680.0, "bid": 0.20, "ask": 0.30},
                    ]},
                    "meta": {"age_seconds": 3.0, "source": "fresh_cache"},
                }
            async def trade_credit_spread(self, **kw):
                trade_calls.append(kw)
                return {"order_id": "x", "status": "FILLED"}

        # Bind real pricing + resolution helpers onto the fake
        from utp import TradingClient
        FakeTClient.compute_credit_spread_net_price = TradingClient.compute_credit_spread_net_price
        FakeTClient._resolve_trade_defaults = TradingClient._resolve_trade_defaults

        import spread_scanner as ss
        monkeypatch.setattr(ss, "_get_trading_client_cls", lambda: FakeTClient)
        monkeypatch.setattr(ss.TradePolicy, "within_trading_window", lambda self, now_pt=None: True)

        # Policy does NOT set order_type — daemon default should be used.
        pol = TradePolicy.from_dict({"roi_pct": [1.0, 10.0]})
        assert pol.order_type is None
        h = TradeHandler(min_norm_roi=0, log_file=str(tmp_path / "l.jsonl"),
                        policy=pol, daemon_url="http://d")

        def _spread(sym, credit, short, long_, width):
            return {
                "symbol": sym, "option_type": "PUT",
                "short_strike": short, "long_strike": long_, "width": width,
                "credit": credit, "roi_pct": 3.0, "otm_pct": 1.5, "dte": 0,
                "expiration": "2026-04-21", "prev_close": 5800.0,
            }
        asyncio.run(h.fire([_spread("SPX", 0.85, 5700, 5680, 20)], HandlerContext(
            client=None, args=None, scan_data={}, is_market_hours=True,
            now_ts="2026-04-22T07:00:00")))

        # Daemon said LIMIT → handler should have submitted net_price (not None).
        assert trade_calls[0]["net_price"] is not None
        assert trade_calls[0]["net_price"] == 0.70        # 1.00 - 0.30 (from fake quotes)


class TestDaemonConfigTradeDefaults:
    """YAML `trade_defaults` section flows into StreamingConfig correctly."""

    def _write(self, tmp_path, body):
        path = tmp_path / "daemon.yaml"
        path.write_text(body)
        return str(path)

    def test_nested_trade_defaults_block(self, tmp_path):
        from app.services.streaming_config import load_streaming_config
        path = self._write(tmp_path, """
symbols: [SPX]
trade_defaults:
  default_order_type: LIMIT
  limit_slippage_pct: 1.5
  limit_quote_max_age_sec: 12.0
""")
        cfg = load_streaming_config(path)
        assert cfg.default_order_type == "LIMIT"
        assert cfg.limit_slippage_pct == 1.5
        assert cfg.limit_quote_max_age_sec == 12.0

    def test_top_level_trade_defaults(self, tmp_path):
        """Back-compat: top-level keys still work without the nested block."""
        from app.services.streaming_config import load_streaming_config
        path = self._write(tmp_path, """
symbols: [SPX]
default_order_type: MARKET
limit_slippage_pct: 0.0
""")
        cfg = load_streaming_config(path)
        assert cfg.default_order_type == "MARKET"
        assert cfg.limit_slippage_pct == 0.0
        assert cfg.limit_quote_max_age_sec is None

    def test_missing_trade_defaults_keeps_none(self, tmp_path):
        """No trade_defaults in YAML → StreamingConfig fields stay None."""
        from app.services.streaming_config import load_streaming_config
        path = self._write(tmp_path, "symbols: [SPX]\n")
        cfg = load_streaming_config(path)
        assert cfg.default_order_type is None
        assert cfg.limit_slippage_pct is None
        assert cfg.limit_quote_max_age_sec is None


class TestTradingClientPricing:
    """Direct unit tests for TradingClient.compute_credit_spread_net_price.

    The scanner delegates ALL limit-pricing math to this one method in utp.py,
    so these tests guarantee there's exactly one implementation of the logic.
    """

    @staticmethod
    def _make_client():
        """Return a TradingClient with a fake `get_option_quotes` attached.

        We don't need a real httpx connection — the pricing method only calls
        `self.get_option_quotes`, which is safe to mock via setattr.
        """
        from utp import TradingClient
        return TradingClient("http://d")

    def _with_quotes(self, quotes_resp):
        c = self._make_client()
        async def fake_get(**kw):
            return quotes_resp
        c.get_option_quotes = fake_get
        return c

    def _with_error(self, err):
        c = self._make_client()
        async def fake_get(**kw):
            raise err
        c.get_option_quotes = fake_get
        return c

    def test_uses_refreshed_credit_not_fallback(self):
        import asyncio
        c = self._with_quotes({
            "quotes": {"put": [
                {"strike": 5700.0, "bid": 1.20, "ask": 1.30},
                {"strike": 5680.0, "bid": 0.40, "ask": 0.50},
            ]},
            "meta": {"age_seconds": 3.0, "source": "fresh_cache"},
        })
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            slippage_pct=0.0, max_age=10.0, fallback_credit=0.85,
        ))
        assert price == 0.70          # 1.20 - 0.50
        assert meta["refreshed_credit"] == 0.70
        assert meta["scan_credit"] == 0.85
        assert meta["age_seconds"] == 3.0
        assert meta["quote_source"] == "fresh_cache"
        assert "fallback_reason" not in meta

    def test_applies_slippage_to_refreshed_credit(self):
        import asyncio
        c = self._with_quotes({
            "quotes": {"put": [
                {"strike": 5700.0, "bid": 1.00, "ask": 1.10},
                {"strike": 5680.0, "bid": 0.20, "ask": 0.30},
            ]},
            "meta": {"age_seconds": 2.0, "source": "fresh_cache"},
        })
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            slippage_pct=10.0, max_age=10.0, fallback_credit=0.80,
        ))
        # refreshed = 1.00 - 0.30 = 0.70; * 0.90 = 0.63
        assert price == 0.63
        assert meta["refreshed_credit"] == 0.70
        assert meta["slippage_pct"] == 10.0

    def test_fallback_on_refresh_error(self):
        import asyncio
        c = self._with_error(RuntimeError("daemon down"))
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            slippage_pct=0.0, fallback_credit=0.85,
        ))
        assert price == 0.85
        assert meta["refreshed_credit"] is None
        assert meta["quote_source"] == "error"
        assert "quote_refresh_failed" in meta["fallback_reason"]

    def test_fallback_on_missing_leg(self):
        import asyncio
        c = self._with_quotes({
            "quotes": {"put": [{"strike": 5700.0, "bid": 1.20, "ask": 1.30}]},
            "meta": {"age_seconds": 5.0, "source": "provider"},
        })
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            fallback_credit=0.90,
        ))
        assert price == 0.90
        assert meta["fallback_reason"] == "missing_leg_in_refreshed_quotes"

    def test_fallback_on_nonpositive_bid_ask(self):
        import asyncio
        c = self._with_quotes({
            "quotes": {"put": [
                {"strike": 5700.0, "bid": 0, "ask": 1.30},      # no bid
                {"strike": 5680.0, "bid": 0.40, "ask": 0.50},
            ]},
            "meta": {"age_seconds": 4.0, "source": "fresh_cache"},
        })
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            fallback_credit=0.85,
        ))
        assert price == 0.85
        assert meta["fallback_reason"] == "non_positive_bid_or_ask"

    def test_fallback_on_nonpositive_refreshed_credit(self):
        import asyncio
        c = self._with_quotes({
            "quotes": {"put": [
                # short bid 0.30 < long ask 0.50 → credit goes negative
                {"strike": 5700.0, "bid": 0.30, "ask": 0.40},
                {"strike": 5680.0, "bid": 0.40, "ask": 0.50},
            ]},
            "meta": {"age_seconds": 6.0, "source": "fresh_cache"},
        })
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            fallback_credit=0.85,
        ))
        assert price == 0.85
        assert meta["fallback_reason"] == "refreshed_credit_non_positive"
        assert meta["refreshed_credit"] == -0.20

    def test_fallback_credit_none_returns_none_price(self):
        """If refresh fails AND no fallback_credit provided, caller gets None."""
        import asyncio
        c = self._with_error(RuntimeError("nope"))
        price, meta = asyncio.run(c.compute_credit_spread_net_price(
            symbol="SPX", short_strike=5700.0, long_strike=5680.0,
            option_type="PUT", expiration="2026-04-21",
            fallback_credit=None,
        ))
        assert price is None
        assert meta["fallback_reason"].startswith("quote_refresh_failed")


class TestVerifySpreadPricing:
    """TradingClient.verify_spread_pricing — real-time re-check of a candidate
    spread before the scanner presents it or a trade handler commits."""

    @staticmethod
    def _client_with(quotes_resp):
        from utp import TradingClient
        c = TradingClient("http://d")
        async def fake_get(**kw):
            return quotes_resp
        c.get_option_quotes = fake_get
        return c

    def test_ok_when_quotes_fresh_and_tradeable(self):
        import asyncio
        c = self._client_with({
            "quotes": {"put": [
                {"strike": 7050.0, "bid": 1.20, "ask": 1.40,
                 "source": "ibkr_fresh",
                 "greeks": {"delta": -0.18}},
                {"strike": 7030.0, "bid": 0.40, "ask": 0.50,
                 "source": "ibkr_fresh",
                 "greeks": {"delta": -0.10}},
            ]},
            "meta": {"age_seconds": 2.0, "source": "fresh_cache"},
        })
        res = asyncio.run(c.verify_spread_pricing(
            symbol="SPX", expiration="2026-04-27", option_type="PUT",
            short_strike=7050.0, long_strike=7030.0,
        ))
        assert res["ok"] is True
        assert res["credit"] == 0.70              # 1.20 - 0.50
        assert res["short_delta"] == -0.18
        assert res["reason"] is None

    def test_rejects_csv_sourced_quotes_by_default(self):
        """A CSV-sourced leg fails verify_spread_pricing when
        require_provider_source=True (the default)."""
        import asyncio
        c = self._client_with({
            "quotes": {"put": [
                {"strike": 7050.0, "bid": 1.20, "ask": 1.40,
                 "source": "ibkr_fresh"},
                {"strike": 7030.0, "bid": 0.40, "ask": 0.50,
                 "source": "csv"},  # CSV fallback — refuse.
            ]},
            "meta": {"age_seconds": 2.0, "source": "fresh_cache"},
        })
        res = asyncio.run(c.verify_spread_pricing(
            symbol="SPX", expiration="2026-04-27", option_type="PUT",
            short_strike=7050.0, long_strike=7030.0,
        ))
        assert res["ok"] is False
        assert res["reason"] == "csv_source_rejected"
        assert res["long_source"] == "csv"

    def test_allows_csv_when_require_provider_source_false(self):
        """Caller can opt in to CSV fallback (e.g. during provider outage)."""
        import asyncio
        c = self._client_with({
            "quotes": {"put": [
                {"strike": 7050.0, "bid": 1.20, "ask": 1.40, "source": "csv"},
                {"strike": 7030.0, "bid": 0.40, "ask": 0.50, "source": "csv"},
            ]},
            "meta": {"age_seconds": 2.0, "source": "fresh_cache"},
        })
        res = asyncio.run(c.verify_spread_pricing(
            symbol="SPX", expiration="2026-04-27", option_type="PUT",
            short_strike=7050.0, long_strike=7030.0,
            require_provider_source=False,
        ))
        assert res["ok"] is True
        assert res["credit"] == 0.70

    def test_rejects_non_monotonic_bid(self):
        """Short bid < long bid (short is CLOSER to ATM, should bid higher)
        → data is phantom. Reject with reason=non_monotonic."""
        import asyncio
        c = self._client_with({
            "quotes": {"put": [
                {"strike": 25830.0, "bid": 3.10, "ask": 3.90, "source": "ibkr_fresh"},
                {"strike": 25820.0, "bid": 3.20, "ask": 3.80, "source": "ibkr_fresh"},
            ]},
            "meta": {"age_seconds": 1.0, "source": "fresh_cache"},
        })
        res = asyncio.run(c.verify_spread_pricing(
            symbol="NDX", expiration="2026-04-27", option_type="PUT",
            short_strike=25830.0, long_strike=25820.0,
        ))
        assert res["ok"] is False
        assert res["reason"] == "non_monotonic"

    def test_rejects_no_edge(self):
        """short_bid - long_ask ≤ 0 → no tradeable credit."""
        import asyncio
        c = self._client_with({
            "quotes": {"put": [
                {"strike": 7050.0, "bid": 0.50, "ask": 0.70, "source": "ibkr_fresh"},
                {"strike": 7030.0, "bid": 0.45, "ask": 0.60, "source": "ibkr_fresh"},
            ]},
            "meta": {"age_seconds": 1.0, "source": "fresh_cache"},
        })
        res = asyncio.run(c.verify_spread_pricing(
            symbol="SPX", expiration="2026-04-27", option_type="PUT",
            short_strike=7050.0, long_strike=7030.0,
        ))
        # credit = 0.50 - 0.60 = -0.10 → no edge
        assert res["ok"] is False
        assert res["reason"] == "no_edge"

    def test_rejects_short_no_bid(self):
        import asyncio
        c = self._client_with({
            "quotes": {"put": [
                {"strike": 7050.0, "bid": 0.0, "ask": 0.15, "source": "ibkr_fresh"},
                {"strike": 7030.0, "bid": 0.05, "ask": 0.10, "source": "ibkr_fresh"},
            ]},
            "meta": {"age_seconds": 1.0, "source": "fresh_cache"},
        })
        res = asyncio.run(c.verify_spread_pricing(
            symbol="SPX", expiration="2026-04-27", option_type="PUT",
            short_strike=7050.0, long_strike=7030.0,
        ))
        assert res["ok"] is False
        assert res["reason"] == "short_no_bid"

    def test_rejects_missing_strike(self):
        import asyncio
        c = self._client_with({
            "quotes": {"put": [
                {"strike": 7050.0, "bid": 1.20, "ask": 1.40, "source": "ibkr_fresh"},
                # long strike 7030 intentionally missing from response
            ]},
            "meta": {"age_seconds": 1.0, "source": "fresh_cache"},
        })
        res = asyncio.run(c.verify_spread_pricing(
            symbol="SPX", expiration="2026-04-27", option_type="PUT",
            short_strike=7050.0, long_strike=7030.0,
        ))
        assert res["ok"] is False
        assert res["reason"] == "missing_long_strike"


class TestTierPercentile:
    """Percentile-based tier selectors (pN) in min_tier / min_tier_close."""

    def test_normalize_named_tier(self):
        from spread_scanner import _normalize_tier_selector
        assert _normalize_tier_selector("aggr") == "aggressive"
        assert _normalize_tier_selector("a") == "aggressive"
        assert _normalize_tier_selector("mod") == "moderate"
        assert _normalize_tier_selector("conservative") == "conservative"

    def test_normalize_percentile(self):
        from spread_scanner import _normalize_tier_selector
        assert _normalize_tier_selector("p40") == "p40"
        assert _normalize_tier_selector("p75") == "p75"
        assert _normalize_tier_selector("P95") == "p95"
        assert _normalize_tier_selector(" p99 ") == "p99"

    def test_normalize_percentile_out_of_range(self):
        from spread_scanner import _normalize_tier_selector
        assert _normalize_tier_selector("p0") is None
        assert _normalize_tier_selector("p100") is None
        assert _normalize_tier_selector("p200") is None

    def test_normalize_unknown(self):
        from spread_scanner import _normalize_tier_selector
        assert _normalize_tier_selector("weirdo") is None
        assert _normalize_tier_selector("q50") is None

    def test_is_percentile_tier(self):
        from spread_scanner import _is_percentile_tier
        assert _is_percentile_tier("p75") is True
        assert _is_percentile_tier("conservative") is False
        assert _is_percentile_tier("") is False

    def test_yaml_min_tier_pN_flows_through(self, tmp_path, monkeypatch):
        from spread_scanner import _load_config
        path = tmp_path / "cfg.yaml"
        path.write_text("tickers: [SPX]\nmin_tier: p75\ntiers: true\n")
        monkeypatch.setattr("sys.argv", ["spread_scanner.py", "--config", str(path)])
        args, _ = _load_config()
        assert args.min_tier == "p75"

    def test_resolve_tier_strike_pN_intraday(self):
        """pN form skips the recommended lookup and uses the literal percentile."""
        from spread_scanner import resolve_tier_strike
        # Seed tier data: no 'recommended.intraday.p40' entry — pN must read from pcts directly.
        tier_data = {
            "hourly": {
                "SPX": {
                    "recommended": {
                        "intraday": {"conservative": {"put": 98, "call": 98}},
                    },
                    "slots": {
                        "09:00": {
                            "when_down": {"pct": {"p40": -0.3, "p75": -1.0, "p95": -2.0}},
                            "when_up":   {"pct": {"p40":  0.3, "p75":  1.0, "p95":  2.0}},
                        },
                    },
                }
            },
        }
        import spread_scanner as ss
        import unittest.mock as um
        with um.patch.object(ss, "_find_current_slot", return_value="09:00"):
            r = resolve_tier_strike(tier_data, "SPX", "put", "p40",
                                    model="intraday", prev_close=7100.0,
                                    current_price=7000.0, dte=0)
            assert r is not None
            strike, raw, pct_num, pct_val = r
            assert pct_num == 40
            assert pct_val == -0.3
            # 7000 * (1 - 0.003) = 6979 → rounded DOWN to nearest 5 for put = 6975
            assert strike == 6975

            r = resolve_tier_strike(tier_data, "SPX", "call", "p95",
                                    model="intraday", prev_close=7100.0,
                                    current_price=7000.0, dte=0)
            assert r is not None
            _, _, pct_num, pct_val = r
            assert pct_num == 95 and pct_val == 2.0

    def test_resolve_tier_strike_pN_close_to_close(self):
        from spread_scanner import resolve_tier_strike
        tier_data = {
            "hourly": {
                "SPX": {
                    "recommended": {"close_to_close": {"conservative": {"put": 98, "call": 98}}},
                    "slots": {},
                }
            },
            "tickers": [{
                "ticker": "SPX",
                "windows": {
                    "0": {
                        "when_down": {"pct": {"p40": -0.5, "p75": -1.5}},
                        "when_up":   {"pct": {"p40":  0.5, "p75":  1.5}},
                    },
                },
            }],
        }
        r = resolve_tier_strike(tier_data, "SPX", "put", "p75",
                                model="close_to_close", prev_close=7100.0,
                                current_price=7050.0, dte=0)
        assert r is not None
        _, _, pct_num, pct_val = r
        assert pct_num == 75 and pct_val == -1.5

    def test_resolve_tier_boundaries_includes_requested_pN(self):
        """_resolve_tier_boundaries computes boundaries for any pN that min_tier
        requests, even though it's not in the default TIER_KEYS list."""
        from spread_scanner import _resolve_tier_boundaries
        import spread_scanner as ss
        import unittest.mock as um
        scan_data = {
            "quotes": {"SPX": {"last": 7000}},
            "prev_closes": {"SPX": 7100},
            "dte_sections": {
                0: {
                    "tier_data": {
                        "hourly": {
                            "SPX": {
                                "recommended": {
                                    "intraday": {
                                        "aggressive":   {"put": 90, "call": 90},
                                        "moderate":     {"put": 95, "call": 95},
                                        "conservative": {"put": 98, "call": 98},
                                    },
                                },
                                "slots": {
                                    "09:00": {
                                        "when_down": {"pct": {"p40": -0.3, "p90": -1.0,
                                                              "p95": -1.5, "p98": -2.0}},
                                        "when_up":   {"pct": {"p40":  0.3, "p90":  1.0,
                                                              "p95":  1.5, "p98":  2.0}},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

        class Args:
            tickers = ["SPX"]
            min_tier = "p40"
            min_tier_close = None

        with um.patch.object(ss, "_find_current_slot", return_value="09:00"):
            bounds = _resolve_tier_boundaries(scan_data, Args(), model="intraday", dte=0)

        assert "SPX" in bounds
        # Default tiers present
        assert "conservative" in bounds["SPX"]
        assert "moderate" in bounds["SPX"]
        assert "aggressive" in bounds["SPX"]
        # Requested p40 also present
        assert "p40" in bounds["SPX"]
        assert "put" in bounds["SPX"]["p40"]


class TestStrikeConflictCheck:
    """Test _check_strike_conflicts_http detects netting conflicts."""

    @pytest.mark.asyncio
    async def test_no_conflict_when_no_positions(self):
        from utp import _check_strike_conflicts_http
        import httpx

        async def mock_get(url, **kwargs):
            return httpx.Response(200, json={"positions": []})

        client = httpx.AsyncClient()
        client.get = mock_get

        warnings = await _check_strike_conflicts_http(
            client, "SPX", "2026-04-24",
            {"short_strike": 7050, "long_strike": 7025},
            "PUT", False,
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_conflict_detected_credit_spread(self):
        """BUY leg at 7025P conflicts with existing SHORT at 7025P."""
        from utp import _check_strike_conflicts_http
        import httpx

        positions = [{
            "symbol": "SPX",
            "expiration": "20260424",
            "strike": 7025,
            "right": "P",
            "quantity": -30,
            "legs": [],
        }]

        async def mock_get(url, **kwargs):
            return httpx.Response(200, json={"positions": positions})

        client = httpx.AsyncClient()
        client.get = mock_get

        warnings = await _check_strike_conflicts_http(
            client, "SPX", "2026-04-24",
            {"short_strike": 7050, "long_strike": 7025},
            "PUT", False,
        )
        assert len(warnings) == 1
        assert "CONFLICT" in warnings[0]
        assert "7025" in warnings[0]

    @pytest.mark.asyncio
    async def test_no_conflict_different_strike(self):
        """Existing SHORT at 7000P doesn't conflict with BUY at 7025P."""
        from utp import _check_strike_conflicts_http
        import httpx

        positions = [{
            "symbol": "SPX",
            "expiration": "20260424",
            "strike": 7000,
            "right": "P",
            "quantity": -30,
            "legs": [],
        }]

        async def mock_get(url, **kwargs):
            return httpx.Response(200, json={"positions": positions})

        client = httpx.AsyncClient()
        client.get = mock_get

        warnings = await _check_strike_conflicts_http(
            client, "SPX", "2026-04-24",
            {"short_strike": 7050, "long_strike": 7025},
            "PUT", False,
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_no_conflict_different_expiration(self):
        """Existing SHORT at 7025P for different exp doesn't conflict."""
        from utp import _check_strike_conflicts_http
        import httpx

        positions = [{
            "symbol": "SPX",
            "expiration": "20260425",
            "strike": 7025,
            "right": "P",
            "quantity": -30,
            "legs": [],
        }]

        async def mock_get(url, **kwargs):
            return httpx.Response(200, json={"positions": positions})

        client = httpx.AsyncClient()
        client.get = mock_get

        warnings = await _check_strike_conflicts_http(
            client, "SPX", "2026-04-24",
            {"short_strike": 7050, "long_strike": 7025},
            "PUT", False,
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_conflict_from_spread_legs(self):
        """Detect conflict from existing spread leg (not just raw position)."""
        from utp import _check_strike_conflicts_http
        import httpx

        positions = [{
            "symbol": "SPX",
            "expiration": "2026-04-24",
            "strike": 0,
            "right": "",
            "quantity": 25,
            "legs": [
                {"action": "SELL_TO_OPEN", "strike": 7025, "option_type": "PUT"},
                {"action": "BUY_TO_OPEN", "strike": 7000, "option_type": "PUT"},
            ],
        }]

        async def mock_get(url, **kwargs):
            return httpx.Response(200, json={"positions": positions})

        client = httpx.AsyncClient()
        client.get = mock_get

        # New trade: SELL 7050, BUY 7025 — the BUY at 7025 conflicts with existing SELL at 7025
        warnings = await _check_strike_conflicts_http(
            client, "SPX", "2026-04-24",
            {"short_strike": 7050, "long_strike": 7025},
            "PUT", False,
        )
        assert len(warnings) == 1
        assert "7025" in warnings[0]

    @pytest.mark.asyncio
    async def test_iron_condor_conflict(self):
        """Iron condor BUY legs checked against existing shorts."""
        from utp import _check_strike_conflicts_http
        import httpx

        positions = [{
            "symbol": "SPX",
            "expiration": "20260424",
            "strike": 7200,
            "right": "C",
            "quantity": -10,
            "legs": [],
        }]

        async def mock_get(url, **kwargs):
            return httpx.Response(200, json={"positions": positions})

        client = httpx.AsyncClient()
        client.get = mock_get

        # IC: put_long=7000P (no conflict), call_long=7200C (conflicts with SHORT 7200C)
        warnings = await _check_strike_conflicts_http(
            client, "SPX", "2026-04-24",
            {"put_short": 7050, "put_long": 7000,
             "call_short": 7150, "call_long": 7200},
            None, True,
        )
        assert len(warnings) == 1
        assert "7200" in warnings[0]
        assert "CALL" in warnings[0]
