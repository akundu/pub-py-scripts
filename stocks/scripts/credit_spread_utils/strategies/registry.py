"""
Strategy registry for looking up strategies by name.

Strategies register themselves at import time using the @StrategyRegistry.register
decorator or by calling StrategyRegistry.register() directly.
"""

from typing import Dict, List, Optional, Type

from .base import BaseStrategy, StrategyConfig


class StrategyRegistry:
    """Registry for looking up strategy classes by name."""

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """Register a strategy class. Can be used as a decorator.

        Args:
            strategy_class: The strategy class to register

        Returns:
            The strategy class (for decorator usage)
        """
        # Instantiate temporarily to get the name
        temp = object.__new__(strategy_class)
        name = strategy_class.name.fget(temp)
        cls._strategies[name] = strategy_class
        return strategy_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """Get a strategy class by name.

        Args:
            name: Strategy name

        Returns:
            Strategy class or None if not found
        """
        return cls._strategies.get(name)

    @classmethod
    def create(
        cls,
        name: str,
        config_dict: dict,
        logger=None,
    ) -> BaseStrategy:
        """Create a strategy instance by name from config.

        Args:
            name: Strategy name
            config_dict: Configuration dictionary
            logger: Optional logger

        Returns:
            Configured strategy instance

        Raises:
            ValueError: If strategy name is not found
        """
        strategy_class = cls.get(name)
        if strategy_class is None:
            available = ', '.join(cls.available())
            raise ValueError(
                f"Unknown strategy '{name}'. Available: {available}"
            )
        return strategy_class.from_json(config_dict, logger=logger)

    @classmethod
    def available(cls) -> List[str]:
        """List all registered strategy names.

        Returns:
            Sorted list of strategy names
        """
        return sorted(cls._strategies.keys())
