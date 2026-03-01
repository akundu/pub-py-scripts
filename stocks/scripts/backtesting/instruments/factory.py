"""InstrumentFactory -- registry for instrument types."""

from typing import Dict, Type

from .base import Instrument


class InstrumentFactory:
    """Maps instrument names to their classes."""

    _registry: Dict[str, Type[Instrument]] = {}

    @classmethod
    def register(cls, name: str, instrument_cls: Type[Instrument]) -> None:
        cls._registry[name] = instrument_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> Instrument:
        if name not in cls._registry:
            raise KeyError(
                f"Unknown instrument '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls):
        return list(cls._registry.keys())
