"""OrchestrationManifest -- loads YAML manifest and builds the instance tree."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

from .algo_instance import AlgoInstance, AlgoInstanceConfig, SubOrchestrator
from .triggers.base import Trigger, TriggerRegistry

# Ensure trigger types are registered
import scripts.backtesting.orchestration.triggers.always       # noqa: F401
import scripts.backtesting.orchestration.triggers.vix_regime   # noqa: F401
import scripts.backtesting.orchestration.triggers.day_of_week  # noqa: F401
import scripts.backtesting.orchestration.triggers.composite    # noqa: F401
import scripts.backtesting.orchestration.triggers.time_window  # noqa: F401


@dataclass
class ExitRulesConfig:
    """Exit rules configuration for interval mode position tracking."""
    profit_target_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    proximity_pct: Optional[float] = None
    time_exit_utc: Optional[str] = None
    roll_check_start_utc: Optional[str] = None


@dataclass
class OrchestrationConfig:
    """Parsed top-level orchestration settings."""
    name: str = "Orchestrator"
    lookback_days: int = 250
    selection_mode: str = "best_score"
    daily_budget: float = 200000
    output_dir: str = "results/orchestrated"
    poll_interval_minutes: int = 10
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Interval mode fields
    phase2_mode: str = "daily"            # "daily" or "interval"
    interval_minutes: int = 5
    interval_budget_mode: str = "decaying"  # "decaying" (remaining / intervals_left)
    top_n: int = 3                        # For top_n selection mode
    interval_budget_cap: Optional[float] = None   # Max risk per interval (flat mode)
    max_risk_per_transaction: Optional[float] = None  # Max risk on any single trade
    scoring_weights: Optional[tuple] = None        # (w_credit, w_volume, w_bidask)
    equity_data: Dict[str, str] = field(default_factory=dict)
    exit_rules: Optional[ExitRulesConfig] = None
    adaptive_budget: Optional[Dict[str, Any]] = None  # Raw dict for AdaptiveBudgetConfig


class OrchestrationManifest:
    """Loads a YAML manifest and builds the instance tree.

    The manifest defines:
    - Global orchestration settings
    - Shared trigger definitions (referenced by name)
    - Groups (sub-orchestrators) and direct instances
    """

    def __init__(self, config: OrchestrationConfig, root_instances: List[AlgoInstance],
                 trigger_defs: Dict[str, Trigger]):
        self.config = config
        self.root_instances = root_instances
        self.trigger_defs = trigger_defs

    @classmethod
    def load(cls, path: str) -> "OrchestrationManifest":
        """Load manifest from YAML file.

        If the YAML has a top-level 'profile' key, loads the unified profile
        (with mode="backtest" overrides) and merges its settings as defaults.
        The orchestration YAML values override the profile values.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        orch = raw.get("orchestration", raw)
        base_dir = str(Path(path).parent)

        # Load unified profile if referenced
        profile_name = raw.get("profile")
        if profile_name:
            try:
                from scripts.live_trading.advisor.profile_loader import load_profile
                profile = load_profile(profile_name, mode="backtest")
                # Merge profile settings into orch as defaults
                # (orch values take precedence over profile values)
                if "daily_budget" not in orch:
                    orch["daily_budget"] = profile.risk.daily_budget
                if "max_risk_per_transaction" not in orch:
                    orch["max_risk_per_transaction"] = profile.risk.max_risk_per_trade
                # Merge adaptive_budget from profile
                if profile.adaptive_budget and profile.adaptive_budget.enabled:
                    profile_ab = {
                        "roi_tier_enabled": profile.adaptive_budget.roi_tier_enabled,
                        "roi_mode": profile.adaptive_budget.roi_mode,
                        "roi_thresholds": profile.adaptive_budget.roi_thresholds,
                        "roi_multipliers": profile.adaptive_budget.roi_multipliers,
                        "roi_max_multiplier": profile.adaptive_budget.roi_max_multiplier,
                        "roi_normalize_dte": profile.adaptive_budget.roi_normalize_dte,
                        "momentum_enabled": profile.adaptive_budget.momentum_enabled,
                        "momentum_threshold": profile.adaptive_budget.momentum_threshold,
                        "reserve_enabled": profile.adaptive_budget.reserve_enabled,
                        "reserve_pct": profile.adaptive_budget.reserve_pct,
                        "contract_scaling_enabled": profile.adaptive_budget.contract_scaling_enabled,
                        "contract_max_multiplier": profile.adaptive_budget.contract_max_multiplier,
                        "dte0_cutoff_utc": profile.adaptive_budget.dte0_cutoff_utc,
                        "vix_budget_multipliers": profile.adaptive_budget.vix_budget_multipliers,
                    }
                    # Orch adaptive_budget overrides profile values
                    existing_ab = orch.get("adaptive_budget", {})
                    merged_ab = {**profile_ab, **existing_ab}
                    orch["adaptive_budget"] = merged_ab
                # Merge exit rules from profile
                if not orch.get("exit_rules"):
                    er = profile.exit_rules
                    orch["exit_rules"] = {
                        "profit_target_pct": er.profit_target_pct,
                        "stop_loss_pct": er.stop_loss_pct,
                        "proximity_pct": er.roll_proximity_pct,
                        "time_exit_utc": er.time_exit_utc,
                        "roll_check_start_utc": er.roll_check_start_utc,
                    }
                # Merge equity data paths from profile
                if not orch.get("equity_data"):
                    eq_dir = profile.providers.equity_csv_dir
                    orch["equity_data"] = {
                        t: f"{eq_dir}/I:{t}" for t in (profile.tickers or [profile.ticker])
                    }
                logger.info(f"Loaded unified profile '{profile_name}' with backtest overrides")
            except Exception as e:
                logger.warning(f"Could not load profile '{profile_name}': {e}")

        # Parse exit rules
        exit_rules_data = orch.get("exit_rules", {})
        exit_rules = None
        if exit_rules_data:
            exit_rules = ExitRulesConfig(
                profit_target_pct=exit_rules_data.get("profit_target_pct"),
                stop_loss_pct=exit_rules_data.get("stop_loss_pct"),
                proximity_pct=exit_rules_data.get("proximity_pct"),
                time_exit_utc=exit_rules_data.get("time_exit_utc"),
                roll_check_start_utc=exit_rules_data.get("roll_check_start_utc"),
            )

        # Parse global config
        config = OrchestrationConfig(
            name=orch.get("name", "Orchestrator"),
            lookback_days=orch.get("lookback_days", 250),
            selection_mode=orch.get("selection_mode", "best_score"),
            daily_budget=orch.get("daily_budget", 200000),
            output_dir=orch.get("output_dir", "results/orchestrated"),
            poll_interval_minutes=orch.get("poll_interval_minutes", 10),
            start_date=orch.get("start_date"),
            end_date=orch.get("end_date"),
            phase2_mode=orch.get("phase2_mode", "daily"),
            interval_minutes=orch.get("interval_minutes", 5),
            interval_budget_mode=orch.get("interval_budget_mode", "decaying"),
            top_n=orch.get("top_n", 3),
            interval_budget_cap=orch.get("interval_budget_cap"),
            max_risk_per_transaction=orch.get("max_risk_per_transaction"),
            scoring_weights=tuple(orch["scoring_weights"]) if orch.get("scoring_weights") else None,
            equity_data=orch.get("equity_data", {}),
            exit_rules=exit_rules,
            adaptive_budget=orch.get("adaptive_budget"),
        )

        # Parse shared trigger definitions
        trigger_defs = cls._parse_triggers(orch.get("triggers", {}))

        # Parse groups (sub-orchestrators)
        root_instances: List[AlgoInstance] = []
        for group_data in orch.get("groups", []):
            sub_orch = cls._parse_group(group_data, trigger_defs, base_dir)
            root_instances.append(sub_orch)

        # Parse direct (ungrouped) instances
        for inst_data in orch.get("instances", []):
            instance = cls._parse_instance(inst_data, trigger_defs, base_dir)
            root_instances.append(instance)

        return cls(config=config, root_instances=root_instances,
                   trigger_defs=trigger_defs)

    @classmethod
    def _parse_triggers(cls, trigger_data: Dict[str, Any]) -> Dict[str, Trigger]:
        """Parse shared trigger definitions from manifest."""
        triggers: Dict[str, Trigger] = {}
        for name, tdef in trigger_data.items():
            trigger_type = tdef.get("type", name)
            params = tdef.get("params", {})
            triggers[name] = TriggerRegistry.create(trigger_type, params)
        return triggers

    @classmethod
    def _parse_instance(cls, data: Dict[str, Any],
                        trigger_defs: Dict[str, Trigger],
                        base_dir: str) -> AlgoInstance:
        """Parse a single algo instance definition."""
        config_path = data.get("config", "")
        # Resolve relative config paths
        if config_path and not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)

        # Resolve trigger references
        trigger_names = data.get("triggers", [])
        trigger_mode = data.get("trigger_mode", "any")
        triggers = cls._resolve_triggers(trigger_names, trigger_defs)

        inst_config = AlgoInstanceConfig(
            algo_name=data.get("algo", ""),
            instance_id=data.get("id", data.get("algo", "unknown")),
            config_path=config_path,
            overrides=data.get("overrides", {}),
            triggers=triggers,
            trigger_mode=trigger_mode,
            priority=data.get("priority", 5),
            budget_share=data.get("budget_share", 1.0),
            enabled=data.get("enabled", True),
        )

        return AlgoInstance(inst_config)

    @classmethod
    def _parse_group(cls, data: Dict[str, Any],
                     trigger_defs: Dict[str, Trigger],
                     base_dir: str) -> SubOrchestrator:
        """Parse a group (sub-orchestrator) with child instances."""
        group_name = data.get("name", "group")
        selection_mode = data.get("selection_mode", "best_score")
        budget_share = data.get("budget_share", 1.0)

        # Parse group triggers
        group_trigger_names = data.get("triggers", [])
        group_triggers = cls._resolve_triggers(group_trigger_names, trigger_defs)

        # Parse children
        children: List[AlgoInstance] = []
        for inst_data in data.get("instances", []):
            child = cls._parse_instance(inst_data, trigger_defs, base_dir)
            children.append(child)

        # Sub-groups (recursive)
        for sub_group_data in data.get("groups", []):
            sub = cls._parse_group(sub_group_data, trigger_defs, base_dir)
            children.append(sub)

        group_config = AlgoInstanceConfig(
            algo_name=f"group:{group_name}",
            instance_id=f"group:{group_name}",
            config_path="",
            triggers=group_triggers,
            priority=data.get("priority", 5),
            budget_share=budget_share,
        )

        return SubOrchestrator(
            config=group_config,
            children=children,
            selection_mode=selection_mode,
        )

    @classmethod
    def _resolve_triggers(cls, names: List[str],
                          trigger_defs: Dict[str, Trigger]) -> List[Trigger]:
        """Resolve trigger names to Trigger instances."""
        triggers = []
        for name in names:
            if name in trigger_defs:
                triggers.append(trigger_defs[name])
            else:
                # Try creating from registry directly
                try:
                    triggers.append(TriggerRegistry.create(name))
                except KeyError:
                    raise ValueError(
                        f"Unknown trigger reference: {name!r}. "
                        f"Available: {list(trigger_defs.keys())}"
                    )
        return triggers

    def get_all_leaf_instances(self) -> List[AlgoInstance]:
        """Get all leaf (non-group) instances for Phase 1 execution."""
        leaves = []
        for inst in self.root_instances:
            if isinstance(inst, SubOrchestrator):
                leaves.extend(inst.all_children)
            else:
                leaves.append(inst)
        return leaves

    def get_instance_by_id(self, instance_id: str) -> Optional[AlgoInstance]:
        """Find an instance by its ID."""
        for inst in self.get_all_leaf_instances():
            if inst.instance_id == instance_id:
                return inst
        return None

    def get_group_by_name(self, name: str) -> Optional[SubOrchestrator]:
        """Find a group by name."""
        for inst in self.root_instances:
            if isinstance(inst, SubOrchestrator) and inst.instance_id == f"group:{name}":
                return inst
        return None

    def print_tree(self, indent: int = 0) -> str:
        """Pretty-print the instance tree."""
        lines = [f"Orchestrator: {self.config.name}"]
        lines.append(f"  Selection: {self.config.selection_mode}, "
                      f"Budget: ${self.config.daily_budget:,.0f}")
        if self.config.phase2_mode == "interval":
            lines.append(f"  Mode: interval ({self.config.interval_minutes}min), "
                          f"top_n: {self.config.top_n}")

        for inst in self.root_instances:
            lines.extend(self._tree_lines(inst, indent=2))

        return "\n".join(lines)

    def _tree_lines(self, inst: AlgoInstance, indent: int) -> List[str]:
        prefix = " " * indent
        lines = []
        if isinstance(inst, SubOrchestrator):
            lines.append(f"{prefix}+ Group: {inst.instance_id} "
                         f"(mode={inst.selection_mode}, "
                         f"budget_share={inst.budget_share:.0%})")
            for child in inst.children:
                lines.extend(self._tree_lines(child, indent + 4))
        else:
            trigger_names = [t.name for t in inst.triggers]
            lines.append(f"{prefix}- {inst.instance_id} "
                         f"(algo={inst.algo_name}, priority={inst.priority}, "
                         f"triggers={trigger_names})")
        return lines
