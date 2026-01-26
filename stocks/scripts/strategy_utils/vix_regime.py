"""
VIX regime analysis for strategy recommendations.

Fetches VIX and VIX1D data from QuestDB and classifies market volatility regime.
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
import logging

# VIX regime thresholds
VIX_THRESHOLDS = {
    'LOW': 15,       # Below 15: Low volatility
    'MEDIUM': 20,    # 15-20: Normal volatility
    'HIGH': 30,      # 20-30: Elevated volatility
    'EXTREME': 30,   # Above 30: Crisis/extreme volatility
}

# VIX ticker symbols (Polygon format)
VIX_TICKER = 'I:VIX'
VIX1D_TICKER = 'I:VIX1D'


def classify_regime(vix_current: float, vix_10d_avg: Optional[float] = None) -> str:
    """
    Classify market volatility regime based on VIX level.

    Args:
        vix_current: Current VIX value
        vix_10d_avg: 10-day average VIX (optional, for trend analysis)

    Returns:
        Regime classification: 'LOW', 'MEDIUM', 'HIGH', 'EXTREME', or 'STABLE'
    """
    if vix_current < VIX_THRESHOLDS['LOW']:
        base_regime = 'LOW'
    elif vix_current < VIX_THRESHOLDS['MEDIUM']:
        base_regime = 'MEDIUM'
    elif vix_current < VIX_THRESHOLDS['HIGH']:
        base_regime = 'HIGH'
    else:
        base_regime = 'EXTREME'

    # Check for stability if we have 10-day average
    if vix_10d_avg is not None:
        # If VIX is close to its 10-day average (within 10%), market is stable
        if abs(vix_current - vix_10d_avg) / vix_10d_avg < 0.10:
            if base_regime == 'LOW':
                return 'STABLE'

    return base_regime


def get_regime_description(regime: str) -> str:
    """Get human-readable description of a VIX regime."""
    descriptions = {
        'LOW': 'Low Volatility - favor longer timeframe parameters',
        'STABLE': 'Stable Market - trust longer timeframe data',
        'MEDIUM': 'Normal Volatility - balanced approach',
        'HIGH': 'Elevated Volatility - favor recent data, reduce exposure',
        'EXTREME': 'Crisis Mode - heavy weighting on recent data, reduce size',
    }
    return descriptions.get(regime, 'Unknown regime')


async def get_vix_data_from_db(db: Any, ticker: str, lookback_days: int = 30) -> Optional[Dict[str, Any]]:
    """
    Fetch VIX data from QuestDB.

    Args:
        db: StockQuestDB instance
        ticker: VIX ticker symbol (e.g., 'I:VIX')
        lookback_days: Number of days to fetch for averaging

    Returns:
        Dict with 'current', 'average_10d', 'data' keys, or None if fetch fails
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Query for daily VIX data
        async with db.connection.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT date, close
                FROM daily_prices
                WHERE ticker = $1
                  AND date >= $2
                  AND date < $3
                ORDER BY date DESC
                """,
                ticker,
                start_date,
                end_date
            )

        if not rows:
            return None

        # Convert to list of dicts
        data = [{'date': row['date'], 'close': float(row['close'])} for row in rows]

        # Get current (most recent) value
        current = data[0]['close']

        # Calculate 10-day average
        recent_10 = data[:10]
        avg_10d = sum(d['close'] for d in recent_10) / len(recent_10) if recent_10 else current

        return {
            'current': current,
            'average_10d': round(avg_10d, 2),
            'data': data,
        }

    except Exception as e:
        logging.warning(f"Failed to fetch VIX data for {ticker}: {e}")
        return None


async def get_vix_regime(db: Any, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Get current VIX metrics and classify market regime.

    Args:
        db: StockQuestDB instance
        logger: Optional logger

    Returns:
        Dict with VIX metrics and regime classification:
        {
            'vix_current': float,
            'vix_10d_avg': float,
            'vix1d_current': float or None,
            'regime': str,
            'regime_description': str,
            'data_available': bool,
        }
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Fetch VIX and VIX1D in parallel
    vix_data, vix1d_data = await asyncio.gather(
        get_vix_data_from_db(db, VIX_TICKER),
        get_vix_data_from_db(db, VIX1D_TICKER),
        return_exceptions=True
    )

    # Handle exceptions
    if isinstance(vix_data, Exception):
        logger.warning(f"VIX fetch error: {vix_data}")
        vix_data = None
    if isinstance(vix1d_data, Exception):
        logger.warning(f"VIX1D fetch error: {vix1d_data}")
        vix1d_data = None

    # Build result
    result = {
        'vix_current': None,
        'vix_10d_avg': None,
        'vix1d_current': None,
        'regime': 'MEDIUM',  # Default to medium if no data
        'regime_description': '',
        'data_available': False,
    }

    if vix_data:
        result['vix_current'] = vix_data['current']
        result['vix_10d_avg'] = vix_data['average_10d']
        result['data_available'] = True

        # Classify regime
        result['regime'] = classify_regime(
            vix_data['current'],
            vix_data['average_10d']
        )

    if vix1d_data:
        result['vix1d_current'] = vix1d_data['current']

    result['regime_description'] = get_regime_description(result['regime'])

    return result


def get_mock_vix_regime() -> Dict[str, Any]:
    """
    Return mock VIX data for testing or when database is unavailable.

    Returns:
        Dict with mock VIX metrics (assumes medium/stable regime)
    """
    return {
        'vix_current': 17.5,
        'vix_10d_avg': 16.8,
        'vix1d_current': 16.2,
        'regime': 'MEDIUM',
        'regime_description': get_regime_description('MEDIUM'),
        'data_available': False,
        'mock_data': True,
    }


def adjust_position_size_for_regime(
    base_risk_cap: float,
    regime: str
) -> float:
    """
    Adjust position size based on VIX regime.

    Note: Per user requirements, VIX primarily adjusts timeframe weights,
    not position sizing. This function is provided for optional use.

    Args:
        base_risk_cap: Base risk cap per trade
        regime: VIX regime classification

    Returns:
        Adjusted risk cap
    """
    adjustments = {
        'LOW': 1.0,      # No adjustment
        'STABLE': 1.0,   # No adjustment
        'MEDIUM': 1.0,   # No adjustment
        'HIGH': 0.75,    # Reduce by 25%
        'EXTREME': 0.50, # Reduce by 50%
    }

    multiplier = adjustments.get(regime, 1.0)
    return base_risk_cap * multiplier
