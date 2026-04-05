"""Trigger ABC, TriggerContext, and TriggerRegistry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Optional, Type


@dataclass
class TriggerContext:
    """Context passed to triggers for evaluation.

    Built once per trading slot by the orchestrator, shared by all triggers.
    Supports both daily (one context per date) and interval modes
    (one context per 5-min interval with intraday fields populated).
    """
    trading_date: date
    day_of_week: int              # 0=Mon, 4=Fri
    vix_regime: Optional[str] = None     # "low", "normal", "high", "extreme"
    vix_close: Optional[float] = None
    vix_percentile_rank: Optional[float] = None
    prev_close: Optional[float] = None
    current_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Intraday fields (populated in interval mode only)
    current_time: Optional[datetime] = None
    intraday_return: Optional[float] = None      # Current price vs day open
    interval_index: Optional[int] = None         # 0-based index within the day
    intervals_remaining: Optional[int] = None    # Intervals left in the day


class Trigger(ABC):
    """Abstract base for orchestration triggers.

    A trigger evaluates whether its conditions are met for a given context.
    Used by AlgoInstance to gate activation before polling the strategy.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    @abstractmethod
    def evaluate(self, context: TriggerContext) -> bool:
        """Return True if this trigger's conditions are met."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class TriggerRegistry:
    """Registry for trigger types."""
    _registry: Dict[str, Type[Trigger]] = {}

    @classmethod
    def register(cls, name: str, trigger_cls: Type[Trigger]) -> None:
        cls._registry[name] = trigger_cls

    @classmethod
    def get(cls, name: str) -> Type[Trigger]:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown trigger type: {name!r}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, params: Optional[Dict[str, Any]] = None) -> Trigger:
        """Create a trigger instance by type name."""
        trigger_cls = cls.get(name)
        return trigger_cls(params=params)

    @classmethod
    def available(cls):
        return list(cls._registry.keys())
