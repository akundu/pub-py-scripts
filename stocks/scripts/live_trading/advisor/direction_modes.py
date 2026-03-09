"""Pluggable directional mode registry for the live advisor.

Each tier specifies a directional mode (e.g., "pursuit", "pursuit_eod") that
determines whether to sell puts or calls based on market conditions. This
module provides an ABC and a registry so new modes can be added without
modifying the evaluator.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, Optional, Type

from .profile_loader import TierDef

logger = logging.getLogger(__name__)


class DirectionalMode(ABC):
    """Base class for directional mode plugins."""

    @abstractmethod
    def get_direction(
        self,
        tier: TierDef,
        current_price: float,
        prev_close: float,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Return "put", "call", or None (skip this tier).

        Args:
            tier: The tier definition being evaluated.
            current_price: Current underlying price.
            prev_close: Previous day's close.
            context: Dict with keys like "eod_state", "tracker", etc.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DIRECTION_REGISTRY: Dict[str, Type[DirectionalMode]] = {}


def register_direction(name: str):
    """Decorator to register a directional mode."""
    def wrapper(cls: Type[DirectionalMode]):
        DIRECTION_REGISTRY[name] = cls
        return cls
    return wrapper


def get_direction_mode(name: str) -> DirectionalMode:
    """Instantiate a registered directional mode by name.

    Raises:
        KeyError: If the mode name is not registered.
    """
    if name not in DIRECTION_REGISTRY:
        raise KeyError(
            f"Unknown directional mode '{name}'. "
            f"Available: {list(DIRECTION_REGISTRY.keys())}"
        )
    return DIRECTION_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Built-in modes
# ---------------------------------------------------------------------------
@register_direction("pursuit")
class PursuitMode(DirectionalMode):
    """Price up -> sell calls (chase upward move), price down -> sell puts."""

    def get_direction(
        self,
        tier: TierDef,
        current_price: float,
        prev_close: float,
        context: Dict[str, Any],
    ) -> Optional[str]:
        if current_price > prev_close:
            return "call"
        elif current_price < prev_close:
            return "put"
        return None  # flat


@register_direction("pursuit_eod")
class PursuitEODMode(DirectionalMode):
    """Direction locked from previous day's close if move > threshold."""

    def get_direction(
        self,
        tier: TierDef,
        current_price: float,
        prev_close: float,
        context: Dict[str, Any],
    ) -> Optional[str]:
        tracker = context.get("tracker")
        if tracker is None:
            return None

        eod_state = tracker.get_eod_state()
        if eod_state.skip_next_day or eod_state.direction is None:
            return None

        # Check if the EOD state was computed yesterday (or earlier)
        today = date.today().isoformat()
        if eod_state.computed_date >= today:
            return None  # Computed today means it's for tomorrow

        return eod_state.direction
