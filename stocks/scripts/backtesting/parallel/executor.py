"""BacktestExecutor -- multiprocessing support for parallel backtests."""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional


class BacktestExecutor:
    """Executes backtest tasks in parallel using multiprocessing."""

    def __init__(
        self,
        num_processes: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        self._num_processes = num_processes or mp.cpu_count()
        self.logger = logger or logging.getLogger(__name__)

    @property
    def num_processes(self) -> int:
        return self._num_processes

    def map(
        self,
        func: Callable,
        items: List[Any],
        **kwargs,
    ) -> List[Any]:
        """Execute func(item) for each item in parallel.

        Args:
            func: Callable that takes a single item.
            items: List of items to process.

        Returns:
            List of results in order.
        """
        if not items:
            return []

        if self._num_processes == 1:
            return [func(item, **kwargs) for item in items]

        results = [None] * len(items)
        with ProcessPoolExecutor(max_workers=self._num_processes) as executor:
            future_to_idx = {}
            for i, item in enumerate(items):
                future = executor.submit(func, item, **kwargs)
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"Task {idx} failed: {e}")
                    results[idx] = None

        return results

    def run_configs(
        self,
        configs: List[Dict],
        run_func: Callable,
    ) -> List[Dict]:
        """Run multiple backtest configurations in parallel.

        Args:
            configs: List of config dictionaries.
            run_func: Function that takes a config and returns results dict.

        Returns:
            List of result dicts.
        """
        return self.map(run_func, configs)
