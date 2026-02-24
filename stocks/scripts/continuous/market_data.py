#!/usr/bin/env python3
"""
Market Data Fetcher for Continuous Mode
"""

import sys
import time as _time
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import pytz

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.stock_db import get_stock_db
from common.financial_data import get_financial_info
from scripts.regime_strategy_selector import detect_vix_regime

# ---------------------------------------------------------------------------
# Price / VIX cache  (60-second TTL)
# ---------------------------------------------------------------------------
_price_cache: Dict[str, Tuple[float, float, float]] = {}   # ticker -> (price, change_pct, timestamp)
_CACHE_TTL = 60  # seconds


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


def _fetch_price(ticker: str) -> Tuple[float, float]:
    """
    Fetch live price and change% for *ticker*.

    Priority: cache → yfinance → stock_db → hardcoded default.
    Returns (current_price, change_pct).
    """
    cache_key = f"price_{ticker}"
    cached = _price_cache.get(cache_key)
    if cached and (_time.time() - cached[2]) < _CACHE_TTL:
        return cached[0], cached[1]

    # --- Primary: yfinance ---
    try:
        import yfinance as yf
        yf_ticker = "^NDX" if ticker.upper() == "NDX" else ticker
        info = yf.Ticker(yf_ticker).fast_info
        current_price = float(info["lastPrice"])
        prev_close = float(info["previousClose"])
        change_pct = ((current_price - prev_close) / prev_close) * 100.0 if prev_close else 0.0
        _price_cache[cache_key] = (current_price, change_pct, _time.time())
        print(f"[market_data] Live NDX price via yfinance: ${current_price:,.2f} ({change_pct:+.2f}%)")
        return current_price, change_pct
    except Exception as e:
        print(f"[market_data] yfinance price fetch failed: {e}")

    # --- Fallback: stock_db ---
    try:
        db = get_stock_db("questdb")
        loop = asyncio.new_event_loop()
        price = loop.run_until_complete(db.get_latest_price(ticker))
        loop.close()
        if price is not None:
            _price_cache[cache_key] = (price, 0.0, _time.time())
            print(f"[market_data] NDX price via stock_db: ${price:,.2f}")
            return price, 0.0
    except Exception as e:
        print(f"[market_data] stock_db price fetch failed: {e}")

    # --- Last resort ---
    print("[market_data] Using hardcoded default price $20,000")
    return 20000.0, 0.0


def _fetch_vix() -> float:
    """
    Fetch live VIX level.

    Priority: cache → yfinance → hardcoded default.
    """
    cache_key = "vix"
    cached = _price_cache.get(cache_key)
    if cached and (_time.time() - cached[2]) < _CACHE_TTL:
        return cached[0]

    try:
        import yfinance as yf
        info = yf.Ticker("^VIX").fast_info
        vix = float(info["lastPrice"])
        _price_cache[cache_key] = (vix, 0.0, _time.time())
        print(f"[market_data] Live VIX via yfinance: {vix:.2f}")
        return vix
    except Exception as e:
        print(f"[market_data] yfinance VIX fetch failed: {e}")

    print("[market_data] Using hardcoded default VIX 15.0")
    return 15.0


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

    # Get current price (cached for 60s)
    current_price, price_change_pct = _fetch_price(ticker)
    current_volume = 0
    avg_volume = 0
    volume_ratio = 1.0

    # Get VIX (cached for 60s)
    vix_level = _fetch_vix()

    vix_regime = detect_vix_regime(vix_level)

    # IV metrics (not yet wired to a live source)
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
        avg_volume_20d=int(avg_volume) if avg_volume else 0,
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
