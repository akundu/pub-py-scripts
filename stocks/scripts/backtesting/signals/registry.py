"""Registry for signal generator classes."""

from typing import Dict, Type

from .base import SignalGenerator


class SignalGeneratorRegistry:
    """Maps signal generator names to their classes."""

    _registry: Dict[str, Type[SignalGenerator]] = {}

    @classmethod
    def register(cls, name: str, generator_cls: Type[SignalGenerator]) -> None:
        cls._registry[name] = generator_cls

    @classmethod
    def get(cls, name: str) -> Type[SignalGenerator]:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown signal generator '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def available(cls):
        return list(cls._registry.keys())
