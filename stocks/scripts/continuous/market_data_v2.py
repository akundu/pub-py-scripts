#!/usr/bin/env python3
"""
Market Data Fetcher for Continuous Mode (V2)

Uses pluggable data providers for flexibility.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, asdict
import pytz

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.regime_strategy_selector import detect_vix_regime
from scripts.continuous.data_providers import (
    DataProvider,
    CSVDataProvider,
    QuestDBProvider,
)
from scripts.continuous.data_providers.composite_provider import CompositeDataProvider


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
    vix1d: Optional[float] = None

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

    # VIX dynamics (rate of change)
    vix_change_5m: Optional[float] = None   # VIX change over last 5 min (absolute)
    vix_change_30m: Optional[float] = None  # VIX change over last 30 min (absolute)
    vix_direction: str = 'stable'           # rising, falling, stable
    vix_velocity: float = 0.0              # Rate of change (points per 5 min)
    vix_term_spread: Optional[float] = None # VIX - VIX1D (positive = near-term stress)

    # Data freshness
    is_stale: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


def create_default_provider(
    csv_base_dir: Optional[Path] = None,
    questdb_config: Optional[str] = None
) -> DataProvider:
    """
    Create default data provider with fallback chain.

    Priority:
      1. QuestDB for VIX/VIX1D (real-time)
      2. CSV files for underlying prices (from option spreads)

    Args:
        csv_base_dir: Base directory for CSV files (default: csv_exports/options)
        questdb_config: QuestDB connection string (default: from env vars)

    Returns:
        Composite data provider
    """
    providers = []

    # QuestDB provider (priority for VIX data)
    if questdb_config is None:
        # Try environment variables
        questdb_config = (
            os.getenv('QUEST_DB_STRING') or
            os.getenv('QUESTDB_CONNECTION_STRING') or
            os.getenv('QUESTDB_URL')
        )

    if questdb_config:
        try:
            providers.append(QuestDBProvider(questdb_config))
            print(f"✓ QuestDB provider initialized")
        except Exception as e:
            print(f"Warning: Could not initialize QuestDB provider: {e}")

    # CSV provider (fallback for prices)
    try:
        csv_provider = CSVDataProvider(csv_base_dir)
        providers.append(csv_provider)
        print(f"✓ CSV provider initialized (dir: {csv_provider.base_dir})")
    except Exception as e:
        print(f"Warning: Could not initialize CSV provider: {e}")

    if not providers:
        raise RuntimeError("No data providers available!")

    return CompositeDataProvider(providers)


def get_current_market_context(
    ticker: str = 'NDX',
    trend: str = 'sideways',
    provider: Optional[DataProvider] = None,
    csv_base_dir: Optional[Path] = None,
    questdb_config: Optional[str] = None
) -> MarketContext:
    """
    Fetch current market context using data providers.

    Args:
        ticker: Ticker symbol
        trend: Market trend (up/down/sideways)
        provider: Data provider (if None, creates default)
        csv_base_dir: CSV directory (used if provider is None)
        questdb_config: QuestDB config (used if provider is None)

    Returns:
        MarketContext object
    """
    # Create provider if not provided
    if provider is None:
        provider = create_default_provider(csv_base_dir, questdb_config)

    now = datetime.now(timezone.utc)
    pst = pytz.timezone('America/Los_Angeles')
    now_pst = now.astimezone(pst)

    # Get market data for ticker
    market_data = provider.get_market_data(ticker)

    if market_data:
        current_price = market_data.current_price or 20000.0
        price_change_pct = market_data.price_change_pct
        volume = market_data.volume or 0
        avg_volume = market_data.avg_volume_20d or 1
        volume_ratio = market_data.volume_ratio
        is_stale = provider.is_stale(ticker, max_age_minutes=5)
    else:
        # Fallback values
        current_price = 20000.0
        price_change_pct = 0.0
        volume = 0
        avg_volume = 1
        volume_ratio = 1.0
        is_stale = True
        print(f"Warning: No market data available for {ticker}")

    # Get VIX data
    vix_data = provider.get_vix_data()
    vix_level = vix_data.get('VIX', 15.0)
    vix1d = vix_data.get('VIX1D')

    if vix_level is None:
        vix_level = 15.0
        print(f"Warning: VIX not available, using default 15.0")

    vix_regime = detect_vix_regime(vix_level)

    # Get VIX dynamics (direction, velocity, term spread)
    vix_dynamics = provider.get_vix_dynamics()

    # Check if market hours (6:30 AM - 1:00 PM PST)
    current_hour = now_pst.hour
    is_market_hours = (6 <= current_hour < 13) and (now_pst.weekday() < 5)

    # Compute term spread from current readings if not in dynamics
    vix_term_spread = vix_dynamics.get('vix_term_spread')
    if vix_term_spread is None and vix1d is not None and vix_level is not None:
        vix_term_spread = vix_level - vix1d

    context = MarketContext(
        timestamp=now.isoformat(),
        ticker=ticker,
        current_price=current_price,
        price_change_pct=price_change_pct,
        vix_level=vix_level,
        vix_regime=vix_regime,
        vix1d=vix1d,
        iv_rank=market_data.iv_rank if market_data else None,
        iv_percentile=market_data.iv_percentile if market_data else None,
        current_volume=volume,
        avg_volume_20d=avg_volume,
        volume_ratio=volume_ratio,
        is_market_hours=is_market_hours,
        current_hour_pst=current_hour,
        trend=trend,
        vix_change_5m=vix_dynamics.get('vix_change_5m'),
        vix_change_30m=vix_dynamics.get('vix_change_30m'),
        vix_direction=vix_dynamics.get('vix_direction', 'stable'),
        vix_velocity=vix_dynamics.get('vix_velocity', 0.0),
        vix_term_spread=vix_term_spread,
        is_stale=is_stale,
    )

    return context


if __name__ == '__main__':
    """Test market data fetching with providers."""
    print("=" * 80)
    print("MARKET DATA PROVIDER TEST")
    print("=" * 80)

    # Test with default providers
    print("\nCreating data providers...")
    provider = create_default_provider()

    print("\nFetching market context for NDX...")
    context = get_current_market_context('NDX', provider=provider)

    print("\n" + "=" * 60)
    print("MARKET CONTEXT")
    print("=" * 60)
    print(f"Timestamp: {context.timestamp}")
    print(f"Ticker: {context.ticker}")
    print(f"Price: ${context.current_price:.2f} ({context.price_change_pct:+.2f}%)")
    print(f"VIX: {context.vix_level:.2f} (Regime: {context.vix_regime.upper()})")
    print(f"VIX1D: {context.vix1d:.2f}" if context.vix1d else "VIX1D: N/A")
    print(f"Volume Ratio: {context.volume_ratio:.2f}x")
    print(f"Market Hours: {'YES' if context.is_market_hours else 'NO'} (Hour: {context.current_hour_pst}:00 PST)")
    print(f"Trend: {context.trend.upper()}")
    print(f"Stale: {'YES' if context.is_stale else 'NO'}")

    # Test VIX data separately
    print("\n" + "=" * 60)
    print("VIX DATA")
    print("=" * 60)
    vix_data = provider.get_vix_data()
    print(f"VIX: {vix_data.get('VIX')}")
    print(f"VIX1D: {vix_data.get('VIX1D')}")

    # Cleanup
    provider.close()
