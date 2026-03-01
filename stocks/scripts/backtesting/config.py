"""Configuration dataclasses and YAML/JSON loader for the backtesting framework."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class InfraConfig:
    """Infrastructure and execution settings."""
    ticker: str = "NDX"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_days: int = 250
    num_processes: int = 0
    output_dir: str = "results/backtest"
    log_level: str = "INFO"
    cache_dir: str = ".backtest_cache"


@dataclass
class ProviderEntry:
    """Single data provider configuration."""
    name: str
    role: str  # "equity", "options", "realtime"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """Supports multiple providers per test."""
    providers: List[ProviderEntry] = field(default_factory=list)

    def get_by_role(self, role: str) -> Optional[ProviderEntry]:
        for p in self.providers:
            if p.role == role:
                return p
        return None

    def get_by_name(self, name: str) -> Optional[ProviderEntry]:
        for p in self.providers:
            if p.name == name:
                return p
        return None


@dataclass
class StrategyConfig:
    """Strategy selection and parameters."""
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetConfig:
    """Budget constraint settings."""
    max_spend_per_transaction: Optional[float] = None
    daily_budget: Optional[float] = None
    gradual_distribution: Optional[Dict[str, Any]] = None


@dataclass
class TradingHoursConfig:
    """Trading hours constraint settings."""
    entry_start: Optional[str] = None
    entry_end: Optional[str] = None
    forced_exit_time: Optional[str] = None


@dataclass
class ExitRulesConfig:
    """Exit rule settings."""
    profit_target_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    time_exit: Optional[str] = None
    mode: str = "first_triggered"


@dataclass
class ConstraintConfig:
    """All constraint settings."""
    budget: Optional[BudgetConfig] = None
    trading_hours: Optional[TradingHoursConfig] = None
    exit_rules: Optional[ExitRulesConfig] = None


@dataclass
class GridSweepConfig:
    """Grid sweep settings."""
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    batch_size: int = 100
    save_intermediate: bool = True


@dataclass
class ReportConfig:
    """Reporting settings."""
    formats: List[str] = field(default_factory=lambda: ["console"])
    metrics: List[str] = field(default_factory=lambda: [
        "win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"
    ])
    grid_sweep: Optional[GridSweepConfig] = None


@dataclass
class BacktestConfig:
    """Top-level backtest configuration."""
    infra: InfraConfig = field(default_factory=InfraConfig)
    providers: ProviderConfig = field(default_factory=ProviderConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        """Build config from a dictionary (parsed YAML/JSON)."""
        infra = InfraConfig(**data.get("infra", {}))

        # Parse providers
        provider_entries = []
        for p in data.get("providers", []):
            provider_entries.append(ProviderEntry(
                name=p["name"],
                role=p["role"],
                params=p.get("params", {}),
            ))
        providers = ProviderConfig(providers=provider_entries)

        # Parse strategy
        strat_data = data.get("strategy", {})
        strategy = StrategyConfig(
            name=strat_data.get("name", ""),
            params=strat_data.get("params", {}),
        )

        # Parse constraints
        constraints_data = data.get("constraints", {})
        budget = None
        if "budget" in constraints_data:
            b = constraints_data["budget"]
            budget = BudgetConfig(
                max_spend_per_transaction=b.get("max_spend_per_transaction"),
                daily_budget=b.get("daily_budget"),
                gradual_distribution=b.get("gradual_distribution"),
            )
        trading_hours = None
        if "trading_hours" in constraints_data:
            th = constraints_data["trading_hours"]
            trading_hours = TradingHoursConfig(
                entry_start=th.get("entry_start"),
                entry_end=th.get("entry_end"),
                forced_exit_time=th.get("forced_exit_time"),
            )
        exit_rules = None
        if "exit_rules" in constraints_data:
            er = constraints_data["exit_rules"]
            exit_rules = ExitRulesConfig(
                profit_target_pct=er.get("profit_target_pct"),
                stop_loss_pct=er.get("stop_loss_pct"),
                time_exit=er.get("time_exit"),
                mode=er.get("mode", "first_triggered"),
            )
        constraints = ConstraintConfig(
            budget=budget,
            trading_hours=trading_hours,
            exit_rules=exit_rules,
        )

        # Parse report
        report_data = data.get("report", {})
        grid_sweep = None
        if "grid_sweep" in report_data:
            gs = report_data["grid_sweep"]
            grid_sweep = GridSweepConfig(
                param_grid=gs.get("param_grid", {}),
                batch_size=gs.get("batch_size", 100),
                save_intermediate=gs.get("save_intermediate", True),
            )
        report = ReportConfig(
            formats=report_data.get("formats", ["console"]),
            metrics=report_data.get("metrics", [
                "win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"
            ]),
            grid_sweep=grid_sweep,
        )

        return cls(
            infra=infra,
            providers=providers,
            strategy=strategy,
            constraints=constraints,
            report=report,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "BacktestConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "BacktestConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str) -> "BacktestConfig":
        """Load config from YAML or JSON based on extension."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif ext == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def deep_set(self, dotted_key: str, value: Any) -> None:
        """Set a nested value using dot notation (e.g., 'strategy.params.percent_beyond')."""
        parts = dotted_key.split(".")
        obj = self
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        final = parts[-1]
        if isinstance(obj, dict):
            obj[final] = value
        else:
            setattr(obj, final, value)
