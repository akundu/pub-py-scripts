"""
Dynamic spread width calculation utilities.

Calculates max spread width based on short strike distance from previous close.
This allows wider spreads for further OTM positions (lower risk = can afford wider spreads).
"""

from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
import json
from pathlib import Path


@dataclass
class DynamicWidthConfig:
    """Configuration for dynamic spread width calculation."""
    mode: str  # "linear", "stepped", "formula"
    base_width: float = 20.0
    slope_factor: float = 1000.0  # For linear: width = base + (distance_pct * slope)
    max_width: Optional[float] = None  # Ceiling (falls back to max_spread_width)
    min_width: float = 5.0  # Floor
    steps: Optional[Dict[str, float]] = None  # For stepped mode: {"0.01": 20, "0.02": 30}
    formula: Optional[str] = None  # For formula mode (advanced)

    @classmethod
    def from_dict(cls, config: dict) -> 'DynamicWidthConfig':
        """Create config from dictionary."""
        # Convert steps keys to strings if they're floats
        steps = config.get('steps')
        if steps and isinstance(steps, dict):
            steps = {str(k): v for k, v in steps.items()}

        return cls(
            mode=config.get('mode', 'linear'),
            base_width=float(config.get('base_width', 20.0)),
            slope_factor=float(config.get('slope_factor', 1000.0)),
            max_width=float(config['max_width']) if config.get('max_width') is not None else None,
            min_width=float(config.get('min_width', 5.0)),
            steps=steps,
            formula=config.get('formula'),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'mode': self.mode,
            'base_width': self.base_width,
            'slope_factor': self.slope_factor,
            'max_width': self.max_width,
            'min_width': self.min_width,
            'steps': self.steps,
            'formula': self.formula,
        }


def calculate_dynamic_width(
    short_strike: float,
    prev_close: float,
    config: DynamicWidthConfig,
    fallback_max: float = 200.0
) -> float:
    """
    Calculate dynamic max spread width based on short strike distance from prev_close.

    Args:
        short_strike: The short leg strike price
        prev_close: Previous closing price (reference)
        config: Dynamic width configuration
        fallback_max: Default max if config.max_width is None

    Returns:
        Calculated max width for this strike distance
    """
    # Calculate distance percentage
    distance_pct = abs(short_strike - prev_close) / prev_close

    # Calculate width based on mode
    if config.mode == "linear":
        width = config.base_width + (distance_pct * config.slope_factor)
    elif config.mode == "stepped":
        width = _calculate_stepped_width(distance_pct, config)
    elif config.mode == "formula":
        width = _calculate_formula_width(distance_pct, config)
    else:
        width = config.base_width

    # Apply floor and ceiling
    max_ceiling = config.max_width if config.max_width is not None else fallback_max
    width = max(config.min_width, min(width, max_ceiling))

    return width


def _calculate_stepped_width(distance_pct: float, config: DynamicWidthConfig) -> float:
    """Calculate width using stepped lookup table."""
    if not config.steps:
        return config.base_width

    # Sort thresholds and find applicable width
    # Convert string keys to float for comparison
    sorted_thresholds = sorted([(float(k), v) for k, v in config.steps.items()])
    width = config.base_width

    for threshold, step_width in sorted_thresholds:
        if distance_pct >= threshold:
            width = step_width
        else:
            break

    return width


def _calculate_formula_width(distance_pct: float, config: DynamicWidthConfig) -> float:
    """Calculate width using custom formula (sandboxed eval)."""
    if not config.formula:
        return config.base_width

    # Safe evaluation with limited namespace
    namespace = {
        'distance_pct': distance_pct,
        'base_width': config.base_width,
        'slope_factor': config.slope_factor,
        'math': math,
        'min': min,
        'max': max,
        'abs': abs,
        'sqrt': math.sqrt,
        'log': math.log,
    }

    try:
        return float(eval(config.formula, {"__builtins__": {}}, namespace))
    except Exception:
        return config.base_width


def parse_dynamic_width_config(value: str) -> Optional[DynamicWidthConfig]:
    """Parse dynamic width config from CLI argument (JSON string or file path)."""
    if not value:
        return None

    try:
        # Try as JSON string first
        config_dict = json.loads(value)
        return DynamicWidthConfig.from_dict(config_dict)
    except json.JSONDecodeError:
        # Try as file path
        path = Path(value)
        if path.exists():
            with open(path, 'r') as f:
                config_dict = json.load(f)
            return DynamicWidthConfig.from_dict(config_dict)

    return None


def format_dynamic_width_info(config: Optional[DynamicWidthConfig]) -> str:
    """Format dynamic width config for display."""
    if config is None:
        return "Fixed (static max_spread_width)"

    if config.mode == "linear":
        return f"Linear: base={config.base_width}, slope={config.slope_factor}"
    elif config.mode == "stepped":
        steps_str = ", ".join(f"{k}:{v}" for k, v in sorted(config.steps.items())) if config.steps else "none"
        return f"Stepped: base={config.base_width}, steps=[{steps_str}]"
    elif config.mode == "formula":
        return f"Formula: {config.formula}"
    else:
        return f"Unknown mode: {config.mode}"
