"""JSON reporter -- writes results to a JSON file."""

import json
import os
from datetime import date, datetime
from typing import Any, Dict

from .base import ReportGenerator


class _DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


class JSONReporter(ReportGenerator):
    """Writes full backtest results to a JSON file."""

    def generate(self, summary: Dict[str, Any], config: Any) -> None:
        output_dir = "results/backtest"
        if hasattr(config, "infra"):
            output_dir = config.infra.output_dir

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "backtest_results.json")

        with open(path, "w") as f:
            json.dump(summary, f, indent=2, cls=_DateEncoder)

        print(f"JSON results written to {path}")
