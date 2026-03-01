"""Tests for backtesting results collector and metrics."""

from datetime import date

import pytest

from scripts.backtesting.results.collector import ResultCollector
from scripts.backtesting.results.metrics import StandardMetrics


class TestStandardMetrics:
    def test_empty_results(self):
        m = StandardMetrics.compute([])
        assert m["total_trades"] == 0
        assert m["win_rate"] == 0.0
        assert m["net_pnl"] == 0.0

    def test_all_wins(self):
        results = [
            {"pnl": 100, "credit": 100, "max_loss": 500},
            {"pnl": 200, "credit": 200, "max_loss": 500},
        ]
        m = StandardMetrics.compute(results)
        assert m["total_trades"] == 2
        assert m["wins"] == 2
        assert m["losses"] == 0
        assert m["win_rate"] == 100.0
        assert m["net_pnl"] == 300.0

    def test_mixed_results(self):
        results = [
            {"pnl": 100, "credit": 100, "max_loss": 500},
            {"pnl": -200, "credit": 50, "max_loss": 500},
            {"pnl": 50, "credit": 50, "max_loss": 500},
        ]
        m = StandardMetrics.compute(results)
        assert m["total_trades"] == 3
        assert m["wins"] == 2
        assert m["losses"] == 1
        assert m["win_rate"] == pytest.approx(66.67, abs=0.01)
        assert m["net_pnl"] == pytest.approx(-50.0)

    def test_profit_factor(self):
        results = [
            {"pnl": 300, "credit": 300, "max_loss": 500},
            {"pnl": -100, "credit": 50, "max_loss": 500},
        ]
        m = StandardMetrics.compute(results)
        assert m["profit_factor"] == pytest.approx(3.0)

    def test_max_drawdown(self):
        results = [
            {"pnl": 100, "credit": 100, "max_loss": 500},
            {"pnl": -300, "credit": 50, "max_loss": 500},
            {"pnl": 50, "credit": 50, "max_loss": 500},
        ]
        m = StandardMetrics.compute(results)
        # Cumulative: 100, -200, -150
        # Peak: 100, drawdown from peak: 0, 300, 250
        assert m["max_drawdown"] == 300.0

    def test_sharpe_ratio(self):
        results = [
            {"pnl": 100, "credit": 100, "max_loss": 500},
            {"pnl": 110, "credit": 110, "max_loss": 500},
            {"pnl": 90, "credit": 90, "max_loss": 500},
        ]
        m = StandardMetrics.compute(results)
        assert m["sharpe"] != 0  # Just verify it computes


class TestResultCollector:
    def test_add_and_count(self):
        c = ResultCollector()
        c.add({"pnl": 100, "credit": 100, "max_loss": 500})
        c.add({"pnl": -50, "credit": 50, "max_loss": 500})
        assert c.count == 2

    def test_add_batch(self):
        c = ResultCollector()
        c.add_batch([
            {"pnl": 100, "credit": 100, "max_loss": 500},
            {"pnl": -50, "credit": 50, "max_loss": 500},
        ])
        assert c.count == 2

    def test_summarize(self):
        c = ResultCollector()
        c.add_batch([
            {"pnl": 100, "credit": 100, "max_loss": 500, "trading_date": date(2026, 1, 5)},
            {"pnl": -50, "credit": 50, "max_loss": 500, "trading_date": date(2026, 1, 5)},
            {"pnl": 200, "credit": 200, "max_loss": 500, "trading_date": date(2026, 1, 6)},
        ])
        summary = c.summarize()
        assert summary["total_trades"] == 3
        assert "metrics" in summary
        assert "daily_breakdown" in summary
        assert "2026-01-05" in summary["daily_breakdown"]
        assert "2026-01-06" in summary["daily_breakdown"]

    def test_clear(self):
        c = ResultCollector()
        c.add({"pnl": 100, "credit": 100, "max_loss": 500})
        c.clear()
        assert c.count == 0
