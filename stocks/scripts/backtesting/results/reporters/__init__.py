"""Report generators."""

from .base import ReportGenerator
from .console import ConsoleReporter
from .csv_reporter import CSVReporter
from .json_reporter import JSONReporter

_REPORTERS = {
    "console": ConsoleReporter,
    "csv": CSVReporter,
    "json": JSONReporter,
}


def get_reporter(name: str) -> ReportGenerator:
    """Get a reporter instance by name."""
    cls = _REPORTERS.get(name)
    if cls is None:
        raise KeyError(f"Unknown reporter '{name}'. Available: {list(_REPORTERS.keys())}")
    return cls()


__all__ = ["ReportGenerator", "get_reporter", "ConsoleReporter", "CSVReporter", "JSONReporter"]
