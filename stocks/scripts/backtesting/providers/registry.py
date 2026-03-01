"""Registry for data provider classes."""

from typing import Dict, Type

from .base import DataProvider


class DataProviderRegistry:
    """Maps provider names to their classes."""

    _registry: Dict[str, Type[DataProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[DataProvider]) -> None:
        cls._registry[name] = provider_cls

    @classmethod
    def get(cls, name: str) -> Type[DataProvider]:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown provider '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys())
