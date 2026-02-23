"""
Band formatting utilities for NDayBand objects.

Provides helpers to display, compare, and convert NDayBand dicts
into a format compatible with the rest of the prediction stack.
"""

from typing import Dict, Optional
from .model import NDayBand, BAND_NAMES


def format_bands(bands: Dict[str, NDayBand], current_price: float) -> str:
    """Return a readable string summary of all bands."""
    if not bands:
        return "  (no bands)"
    lines = []
    for name in BAND_NAMES:
        b = bands.get(name)
        if b is None:
            continue
        lines.append(
            f"  {name}: [{b.lo_price:>10,.1f} – {b.hi_price:>10,.1f}]  "
            f"({b.lo_pct:+.2f}% … {b.hi_pct:+.2f}%)  "
            f"width={b.width_pct:.2f}%  [{b.source}]"
        )
    return "\n".join(lines)


def check_hit(band: NDayBand, actual_close: float) -> bool:
    """True if actual_close falls inside the band."""
    return band.lo_price <= actual_close <= band.hi_price


def band_midpoint(band: NDayBand) -> float:
    return (band.lo_price + band.hi_price) / 2


def band_error_pct(band: NDayBand, actual_close: float, current_price: float) -> float:
    """Signed error as % of current_price: (actual - midpoint) / current_price * 100"""
    mid = band_midpoint(band)
    return (actual_close - mid) / current_price * 100 if current_price else 0.0
