"""Abstract base class for report generators."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ReportGenerator(ABC):
    """Base class for all report generators. Designed for inheritance."""

    @abstractmethod
    def generate(self, summary: Dict[str, Any], config: Any) -> None:
        """Generate a report from backtest summary."""
        ...
