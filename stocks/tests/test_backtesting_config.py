"""Tests for backtesting config module."""

import os
import tempfile

import pytest
import yaml

from scripts.backtesting.config import (
    BacktestConfig,
    BudgetConfig,
    ConstraintConfig,
    ExitRulesConfig,
    GridSweepConfig,
    InfraConfig,
    ProviderConfig,
    ProviderEntry,
    ReportConfig,
    StrategyConfig,
    TradingHoursConfig,
)


class TestInfraConfig:
    def test_defaults(self):
        cfg = InfraConfig()
        assert cfg.ticker == "NDX"
        assert cfg.lookback_days == 250
        assert cfg.num_processes == 0
        assert cfg.log_level == "INFO"

    def test_custom(self):
        cfg = InfraConfig(ticker="SPX", lookback_days=60)
        assert cfg.ticker == "SPX"
        assert cfg.lookback_days == 60


class TestProviderConfig:
    def test_get_by_role(self):
        entries = [
            ProviderEntry(name="csv_equity", role="equity", params={}),
            ProviderEntry(name="csv_options", role="options", params={}),
        ]
        cfg = ProviderConfig(providers=entries)

        assert cfg.get_by_role("equity").name == "csv_equity"
        assert cfg.get_by_role("options").name == "csv_options"
        assert cfg.get_by_role("realtime") is None

    def test_get_by_name(self):
        entries = [ProviderEntry(name="csv_equity", role="equity", params={})]
        cfg = ProviderConfig(providers=entries)
        assert cfg.get_by_name("csv_equity").role == "equity"
        assert cfg.get_by_name("nonexistent") is None


class TestBacktestConfig:
    def test_from_dict_minimal(self):
        data = {
            "infra": {"ticker": "SPX"},
            "strategy": {"name": "zero_dte_credit_spread"},
        }
        cfg = BacktestConfig.from_dict(data)
        assert cfg.infra.ticker == "SPX"
        assert cfg.strategy.name == "zero_dte_credit_spread"

    def test_from_dict_full(self):
        data = {
            "infra": {
                "ticker": "NDX",
                "start_date": "2025-12-01",
                "end_date": "2026-02-28",
                "num_processes": 8,
            },
            "providers": [
                {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
                {"name": "csv_options", "role": "options", "params": {"csv_dir": "options_csv_output"}},
            ],
            "strategy": {
                "name": "zero_dte_credit_spread",
                "params": {"option_types": ["put", "call"]},
            },
            "constraints": {
                "budget": {
                    "max_spend_per_transaction": 20000,
                    "daily_budget": 100000,
                    "gradual_distribution": {"max_amount": 30000, "window_minutes": 60},
                },
                "trading_hours": {
                    "entry_start": "09:45",
                    "entry_end": "15:00",
                    "forced_exit_time": "15:45",
                },
                "exit_rules": {
                    "profit_target_pct": 0.50,
                    "stop_loss_pct": 2.0,
                    "time_exit": "15:30",
                    "mode": "first_triggered",
                },
            },
            "report": {
                "formats": ["console", "csv"],
                "metrics": ["win_rate", "roi"],
            },
        }
        cfg = BacktestConfig.from_dict(data)

        assert cfg.infra.num_processes == 8
        assert len(cfg.providers.providers) == 2
        assert cfg.constraints.budget.daily_budget == 100000
        assert cfg.constraints.trading_hours.entry_start == "09:45"
        assert cfg.constraints.exit_rules.profit_target_pct == 0.50
        assert "csv" in cfg.report.formats

    def test_from_yaml(self):
        data = {
            "infra": {"ticker": "NDX"},
            "strategy": {"name": "test"},
            "providers": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            cfg = BacktestConfig.from_yaml(f.name)
        os.unlink(f.name)

        assert cfg.infra.ticker == "NDX"
        assert cfg.strategy.name == "test"

    def test_load_yaml(self):
        data = {"infra": {"ticker": "SPX"}, "strategy": {"name": "x"}, "providers": []}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            cfg = BacktestConfig.load(f.name)
        os.unlink(f.name)
        assert cfg.infra.ticker == "SPX"

    def test_deep_set(self):
        cfg = BacktestConfig()
        cfg.strategy.name = "test"
        cfg.strategy.params = {"foo": 1}

        cfg.deep_set("strategy.params.foo", 42)
        assert cfg.strategy.params["foo"] == 42

        cfg.deep_set("infra.ticker", "SPX")
        assert cfg.infra.ticker == "SPX"

    def test_to_dict(self):
        cfg = BacktestConfig()
        d = cfg.to_dict()
        assert "infra" in d
        assert "strategy" in d
        assert d["infra"]["ticker"] == "NDX"

    def test_grid_sweep_config(self):
        data = {
            "infra": {"ticker": "NDX"},
            "strategy": {"name": "test"},
            "providers": [],
            "report": {
                "formats": ["console"],
                "grid_sweep": {
                    "param_grid": {
                        "strategy.params.x": [1, 2, 3],
                    },
                    "batch_size": 50,
                },
            },
        }
        cfg = BacktestConfig.from_dict(data)
        assert cfg.report.grid_sweep is not None
        assert cfg.report.grid_sweep.batch_size == 50
        assert len(cfg.report.grid_sweep.param_grid["strategy.params.x"]) == 3
