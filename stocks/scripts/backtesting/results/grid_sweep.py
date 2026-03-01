"""GridSweep -- parameter sweep presentation layer.

Not a strategy. Runs any strategy across parameter combinations and
presents results.
"""

import copy
import itertools
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import BacktestConfig
from ..engine import BacktestEngine


class GridSweepResult:
    """Container for grid sweep results."""

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(self, params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        self.entries.append({"params": params, "metrics": metrics})

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for entry in self.entries:
            row = dict(entry["params"])
            row.update(entry["metrics"])
            rows.append(row)
        return pd.DataFrame(rows)


class GridSweep:
    """Parameter sweep presentation layer.

    Takes a base BacktestConfig + param_grid, generates all combinations,
    runs the backtest engine for each, collects and ranks results.
    """

    def __init__(
        self,
        base_config: BacktestConfig,
        param_grid: Dict[str, List[Any]],
        executor=None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_config = base_config
        self.param_grid = param_grid
        self.executor = executor
        self.logger = logger or logging.getLogger(__name__)

    def generate_configs(self) -> List[BacktestConfig]:
        """Cartesian product of param_grid applied to base_config."""
        if not self.param_grid:
            return [self.base_config]

        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        combos = list(itertools.product(*value_lists))

        configs = []
        for combo in combos:
            config = copy.deepcopy(self.base_config)
            for key, value in zip(keys, combo):
                config.deep_set(key, value)
            configs.append(config)

        return configs

    def run(self) -> GridSweepResult:
        """Run all configs and collect results."""
        configs = self.generate_configs()
        result = GridSweepResult()

        for i, config in enumerate(configs):
            self.logger.info(f"Grid sweep {i + 1}/{len(configs)}")
            try:
                engine = BacktestEngine(config, self.logger)
                summary = engine.run()
                metrics = summary.get("metrics", {})

                # Extract the varying parameters
                params = {}
                for key in self.param_grid:
                    parts = key.split(".")
                    obj = config
                    for part in parts:
                        if isinstance(obj, dict):
                            obj = obj.get(part)
                        else:
                            obj = getattr(obj, part, None)
                    params[key] = obj

                result.add(params, metrics)
            except Exception as e:
                self.logger.error(f"Grid config {i + 1} failed: {e}")

        return result

    def rank(
        self,
        results: GridSweepResult,
        sort_by: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Rank configs by metrics."""
        df = results.to_dataframe()
        if sort_by and len(df) > 0:
            valid_cols = [c for c in sort_by if c in df.columns]
            if valid_cols:
                df = df.sort_values(valid_cols, ascending=False)
        return df.reset_index(drop=True)
