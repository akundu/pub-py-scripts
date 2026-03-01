"""Registry for backtest strategy classes."""

from typing import Dict, Type

from .base import BacktestStrategy


class BacktestStrategyRegistry:
    """Maps strategy names to their classes."""

    _registry: Dict[str, Type[BacktestStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[BacktestStrategy]) -> None:
        cls._registry[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> Type[BacktestStrategy]:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown strategy '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys())
