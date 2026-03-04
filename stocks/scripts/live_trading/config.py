"""Configuration dataclasses and YAML loader for the live trading platform.

Reuses InfraConfig, ProviderConfig, StrategyConfig, ConstraintConfig, and ReportConfig
from the backtesting framework, adding LiveSpecificConfig for live-only settings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from scripts.backtesting.config import (
    BacktestConfig,
    BudgetConfig,
    ConstraintConfig,
    ExitRulesConfig,
    InfraConfig,
    ProviderConfig,
    ProviderEntry,
    ReportConfig,
    StrategyConfig,
    TradingHoursConfig,
)


@dataclass
class LiveSpecificConfig:
    """Settings specific to live/paper trading."""
    mode: str = "paper"                          # "paper" or "live" (future)
    tick_interval_seconds: int = 10
    signal_check_interval_seconds: int = 60
    position_check_interval_seconds: int = 30
    position_db_path: str = "data/live_trading/positions.json"
    journal_path: str = "data/live_trading/journal.jsonl"
    max_positions: int = 10
    session_start_behavior: str = "resume"       # "resume" | "fresh"


@dataclass
class LiveConfig:
    """Top-level live trading configuration.

    Mirrors BacktestConfig structure with an additional `live` section.
    """
    infra: InfraConfig = field(default_factory=InfraConfig)
    providers: ProviderConfig = field(default_factory=ProviderConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    live: LiveSpecificConfig = field(default_factory=LiveSpecificConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiveConfig":
        """Build config from a dictionary (parsed YAML)."""
        # Reuse BacktestConfig parsing for shared sections
        backtest_config = BacktestConfig.from_dict(data)

        # Parse live-specific section
        live_data = data.get("live", {})
        live = LiveSpecificConfig(
            mode=live_data.get("mode", "paper"),
            tick_interval_seconds=live_data.get("tick_interval_seconds", 10),
            signal_check_interval_seconds=live_data.get("signal_check_interval_seconds", 60),
            position_check_interval_seconds=live_data.get("position_check_interval_seconds", 30),
            position_db_path=live_data.get("position_db_path", "data/live_trading/positions.json"),
            journal_path=live_data.get("journal_path", "data/live_trading/journal.jsonl"),
            max_positions=live_data.get("max_positions", 10),
            session_start_behavior=live_data.get("session_start_behavior", "resume"),
        )

        return cls(
            infra=backtest_config.infra,
            providers=backtest_config.providers,
            strategy=backtest_config.strategy,
            constraints=backtest_config.constraints,
            report=backtest_config.report,
            live=live,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "LiveConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_backtest_config(self) -> BacktestConfig:
        """Convert to BacktestConfig for reusing backtesting components."""
        return BacktestConfig(
            infra=self.infra,
            providers=self.providers,
            strategy=self.strategy,
            constraints=self.constraints,
            report=self.report,
        )
