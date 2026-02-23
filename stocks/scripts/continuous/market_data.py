#!/usr/bin/env python3
"""
Market Data Fetcher for Continuous Mode
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import pytz

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.stock_db import get_stock_db
from common.financial_data import get_financial_info
from scripts.regime_strategy_selector import detect_vix_regime


@dataclass
class MarketContext:
    """Current market context."""
    timestamp: str
    ticker: str

    # Price data
    current_price: float
    price_change_pct: float

    # Volatility
    vix_level: float
    vix_regime: str  # very_low, low, medium, high, extreme

    # IV metrics
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None

    # Volume
    current_volume: int = 0
    avg_volume_20d: int = 0
    volume_ratio: float = 1.0

    # Trading hours
    is_market_hours: bool = False
    current_hour_pst: int = 0

    # Trend (manual input for now)
    trend: str = 'sideways'  # up/down/sideways

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


def get_current_market_context(
    ticker: str = 'NDX',
    trend: str = 'sideways'
) -> MarketContext:
    """
    Fetch current market context.

    Args:
        ticker: Ticker symbol
        trend: Market trend (up/down/sideways)

    Returns:
        MarketContext object
    """
    now = datetime.now(timezone.utc)
    pst = pytz.timezone('America/Los_Angeles')
    now_pst = now.astimezone(pst)

    # Get current price
    # TODO: Integrate with live data feed
    # For now, using fallback values - update with actual data source
    try:
        # Placeholder: Fetch from your data source
        # Example: polygon.io, IB API, broker feed, etc.
        current_price = 20000.0  # Update with live price
        price_change_pct = 0.5  # Update with actual change
        current_volume = 1000000
        avg_volume = 950000
        volume_ratio = current_volume / avg_volume

        print(f"Note: Using placeholder price data. Update market_data.py with live data source.")

    except Exception as e:
        print(f"Warning: Could not fetch price data: {e}")
        current_price = 20000.0
        price_change_pct = 0.0
        current_volume = 0
        avg_volume = 0
        volume_ratio = 1.0

    # Get VIX
    # TODO: Integrate with live VIX feed
    try:
        # Placeholder: Fetch from your data source
        vix_level = 14.5  # Update with live VIX

        print(f"Note: Using placeholder VIX={vix_level}. Update market_data.py with live data source.")

    except Exception as e:
        print(f"Warning: Could not fetch VIX: {e}")
        vix_level = 15.0

    vix_regime = detect_vix_regime(vix_level)

    # Get IV metrics
    # TODO: Integrate with IV data source
    try:
        # Placeholder: Fetch from your data source
        iv_rank = None  # Update with live IV rank
        iv_percentile = None  # Update with live IV percentile

        print(f"Note: Using placeholder IV data. Update market_data.py with live data source.")

    except Exception as e:
        print(f"Warning: Could not fetch IV metrics: {e}")
        iv_rank = None
        iv_percentile = None

    # Check if market hours (6:30 AM - 1:00 PM PST)
    current_hour = now_pst.hour
    is_market_hours = (6 <= current_hour < 13) and (now_pst.weekday() < 5)

    context = MarketContext(
        timestamp=now.isoformat(),
        ticker=ticker,
        current_price=current_price,
        price_change_pct=price_change_pct,
        vix_level=vix_level,
        vix_regime=vix_regime,
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        current_volume=current_volume,
        avg_volume_20d=int(avg_volume),
        volume_ratio=volume_ratio,
        is_market_hours=is_market_hours,
        current_hour_pst=current_hour,
        trend=trend,
    )

    return context


if __name__ == '__main__':
    """Test market data fetching."""
    print("Fetching current market context...")
    context = get_current_market_context('NDX')

    print("\n" + "=" * 60)
    print("MARKET CONTEXT")
    print("=" * 60)
    print(f"Timestamp: {context.timestamp}")
    print(f"Ticker: {context.ticker}")
    print(f"Price: ${context.current_price:.2f} ({context.price_change_pct:+.2f}%)")
    print(f"VIX: {context.vix_level:.2f} (Regime: {context.vix_regime.upper()})")
    print(f"IV Rank: {context.iv_rank:.1f}%" if context.iv_rank else "IV Rank: N/A")
    print(f"Volume Ratio: {context.volume_ratio:.2f}x")
    print(f"Market Hours: {'YES' if context.is_market_hours else 'NO'} (Hour: {context.current_hour_pst}:00 PST)")
    print(f"Trend: {context.trend.upper()}")
