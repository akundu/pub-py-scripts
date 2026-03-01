"""Signal generators -- prediction/analysis layer that feeds into strategies."""

from .base import SignalGenerator
from .registry import SignalGeneratorRegistry

__all__ = ["SignalGenerator", "SignalGeneratorRegistry"]
