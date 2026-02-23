#!/usr/bin/env python3
"""
Feature engineering for multi-day ahead predictions (1DTE - 20DTE).

Computes market context features used to condition the historical return distribution:
- Volatility regime (VIX level, percentile, realized vol)
- Price position (vs SMAs, recent range)
- Momentum (recent returns, trend strength)
- Calendar effects (day of week, OPEX, month-end)
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketContext:
    """Market context features for conditioning multi-day predictions."""
    # Volatility features
    vix: Optional[float] = None              # Current VIX level
    vix_percentile: Optional[float] = None   # VIX rank 0-100
    realized_vol_5d: Optional[float] = None  # 5-day realized volatility
    realized_vol_20d: Optional[float] = None # 20-day realized volatility
    vol_regime: str = "medium"               # low/medium/high

    # TIER 1: IV features (forward-looking volatility)
    iv_rank: Optional[float] = None          # IV percentile vs 1yr realized
    iv_percentile: Optional[float] = None    # Current IV rank (0-100)
    iv_term_structure: Optional[float] = None # 30d IV / 90d IV ratio

    # TIER 1: VIX1D features (1-day implied move)
    vix1d: Optional[float] = None            # 1-day implied volatility
    vix1d_percentile: Optional[float] = None # VIX1D historical rank (0-100)

    # TIER 1: Volume features (flow analysis)
    volume_ratio: float = 1.0                # Today's vol / 20-day avg
    volume_spike: bool = False               # True if ratio > 2.0
    volume_trend: float = 0.0                # 5-day volume slope (normalized)

    # TIER 2: Earnings context
    days_to_earnings: Optional[int] = None   # Days until next earnings
    earnings_within_window: bool = False     # Earnings in prediction window

    # TIER 2: Intraday volatility
    gap_size: float = 0.0                    # (Open - Prev Close) / Prev Close %
    intraday_range: float = 0.0              # (High - Low) / Close %

    # Position features
    position_vs_sma20: float = 0.0           # % distance from 20-day SMA
    position_vs_sma50: float = 0.0           # % distance from 50-day SMA
    position_in_10d_range: float = 50.0      # 0-100 percentile in 10-day range
    position_in_20d_range: float = 50.0      # 0-100 percentile in 20-day range
    distance_from_high_20d: float = 0.0      # % below 20-day high
    distance_from_low_20d: float = 0.0       # % above 20-day low

    # Momentum features
    return_1d: float = 0.0                   # 1-day return %
    return_5d: float = 0.0                   # 5-day return %
    return_10d: float = 0.0                  # 10-day return %
    return_20d: float = 0.0                  # 20-day return %
    consecutive_days: int = 0                # Consecutive up (+) or down (-) days
    trend_strength: float = 0.0              # ADX-like metric 0-100

    # Calendar features
    day_of_week: int = 0                     # 0=Mon, 4=Fri
    is_opex_week: bool = False               # Options expiration week
    days_to_month_end: int = 15              # Trading days until month end
    month: int = 1                           # 1-12

    # Derived features
    is_overbought: bool = False              # position_vs_sma20 > 3%
    is_oversold: bool = False                # position_vs_sma20 < -3%
    is_trending: bool = False                # trend_strength > 25

    def to_dict(self) -> Dict:
        """Convert to dict for model input."""
        return {
            # Volatility features
            'vix': self.vix or 15.0,
            'vix_percentile': self.vix_percentile or 50.0,
            'realized_vol_5d': self.realized_vol_5d or 15.0,
            'realized_vol_20d': self.realized_vol_20d or 15.0,
            'vol_regime_low': 1.0 if self.vol_regime == 'low' else 0.0,
            'vol_regime_high': 1.0 if self.vol_regime == 'high' else 0.0,
            # TIER 1: IV features
            'iv_rank': self.iv_rank or 50.0,
            'iv_percentile': self.iv_percentile or 50.0,
            'iv_term_structure': self.iv_term_structure or 1.0,
            # TIER 1: VIX1D features
            'vix1d': self.vix1d or 0.8,
            'vix1d_percentile': self.vix1d_percentile or 50.0,
            # TIER 1: Volume features
            'volume_ratio': self.volume_ratio,
            'volume_spike': 1.0 if self.volume_spike else 0.0,
            'volume_trend': self.volume_trend,
            # TIER 2: Earnings context
            'days_to_earnings': float(self.days_to_earnings or 100),  # Large default = far away
            'earnings_within_window': 1.0 if self.earnings_within_window else 0.0,
            # TIER 2: Intraday volatility
            'gap_size': self.gap_size,
            'intraday_range': self.intraday_range,
            # Position features
            'position_vs_sma20': self.position_vs_sma20,
            'position_vs_sma50': self.position_vs_sma50,
            'position_in_10d_range': self.position_in_10d_range,
            'position_in_20d_range': self.position_in_20d_range,
            # Momentum features
            'return_1d': self.return_1d,
            'return_5d': self.return_5d,
            'return_10d': self.return_10d,
            'return_20d': self.return_20d,
            'consecutive_days': float(self.consecutive_days),
            'trend_strength': self.trend_strength,
            # Calendar features
            'day_of_week': float(self.day_of_week),
            'is_opex_week': 1.0 if self.is_opex_week else 0.0,
            'days_to_month_end': float(self.days_to_month_end),
            'month': float(self.month),
            # Derived features
            'is_overbought': 1.0 if self.is_overbought else 0.0,
            'is_oversold': 1.0 if self.is_oversold else 0.0,
            'is_trending': 1.0 if self.is_trending else 0.0,
        }


def compute_market_context(
    ticker: str,
    current_price: float,
    current_date: date,
    price_history: pd.DataFrame,
    vix_history: Optional[pd.DataFrame] = None,
    vix1d_history: Optional[pd.DataFrame] = None,
    iv_data: Optional[Dict] = None,
) -> MarketContext:
    """Compute all market context features for current conditions.

    Args:
        ticker: Ticker symbol (NDX, SPX)
        current_price: Current/latest price
        current_date: Date of prediction
        price_history: DataFrame with columns ['date', 'close', 'high', 'low', 'volume']
                       sorted by date ascending, includes at least 60 days back
        vix_history: Optional DataFrame with VIX data ['date', 'close']
        vix1d_history: Optional DataFrame with VIX1D data ['date', 'close']
        iv_data: Optional dict with IV metrics {'iv_rank', 'iv_30d', 'iv_90d'}

    Returns:
        MarketContext object with all computed features
    """
    ctx = MarketContext()

    # Ensure price_history is sorted and recent enough
    price_history = price_history.sort_values('date').reset_index(drop=True)

    # Get recent closes as array
    closes = price_history['close'].values
    dates = pd.to_datetime(price_history['date']).dt.date.values

    if len(closes) < 20:
        # Not enough data, return defaults
        return ctx

    # --- Volatility Features ---
    if vix_history is not None and not vix_history.empty:
        vix_history = vix_history.copy()
        vix_history = vix_history.sort_values('date')
        # Convert date column to date objects for comparison
        vix_dates = pd.to_datetime(vix_history['date']).dt.date
        latest_vix_row = vix_history[vix_dates <= current_date].tail(1)
        if not latest_vix_row.empty:
            ctx.vix = float(latest_vix_row.iloc[0]['close'])

            # VIX percentile over lookback
            vix_values = vix_history['close'].values
            if len(vix_values) > 10:
                ctx.vix_percentile = (vix_values < ctx.vix).sum() / len(vix_values) * 100

                # Volatility regime
                if ctx.vix < 14:
                    ctx.vol_regime = "low"
                elif ctx.vix > 22:
                    ctx.vol_regime = "high"
                else:
                    ctx.vol_regime = "medium"

    # Realized volatility (annualized)
    if len(closes) >= 5:
        returns_5d = np.diff(np.log(closes[-6:]))  # last 5 returns
        ctx.realized_vol_5d = np.std(returns_5d) * np.sqrt(252) * 100  # annualized %

    if len(closes) >= 20:
        returns_20d = np.diff(np.log(closes[-21:]))  # last 20 returns
        ctx.realized_vol_20d = np.std(returns_20d) * np.sqrt(252) * 100

    # --- TIER 1: IV Features ---
    if iv_data:
        # IV rank and percentile (forward-looking volatility expectations)
        ctx.iv_rank = iv_data.get('iv_rank')
        ctx.iv_percentile = iv_data.get('iv_90d_rank')  # Use 90d rank as percentile

        # IV term structure (30d IV / 90d IV ratio)
        iv_30d = iv_data.get('iv_30d')
        iv_90d = iv_data.get('iv_90d')
        if iv_30d and iv_90d and iv_90d > 0:
            ctx.iv_term_structure = iv_30d / iv_90d

    # --- TIER 1: VIX1D Features ---
    if vix1d_history is not None and not vix1d_history.empty:
        vix1d_history = vix1d_history.copy()
        vix1d_history = vix1d_history.sort_values('date')
        # Convert date column to date objects for comparison
        vix1d_dates = pd.to_datetime(vix1d_history['date']).dt.date
        latest_vix1d_row = vix1d_history[vix1d_dates <= current_date].tail(1)
        if not latest_vix1d_row.empty:
            ctx.vix1d = float(latest_vix1d_row.iloc[0]['close'])

            # VIX1D percentile over lookback
            vix1d_values = vix1d_history['close'].values
            if len(vix1d_values) > 10:
                ctx.vix1d_percentile = (vix1d_values < ctx.vix1d).sum() / len(vix1d_values) * 100

    # --- TIER 1: Volume Features ---
    if 'volume' in price_history.columns and len(price_history) >= 20:
        volumes = price_history['volume'].values
        current_volume = volumes[-1]

        # Volume ratio: today's volume / 20-day average
        vol_20d_avg = volumes[-20:].mean()
        if vol_20d_avg > 0:
            ctx.volume_ratio = current_volume / vol_20d_avg
            ctx.volume_spike = ctx.volume_ratio > 2.0

        # Volume trend: 5-day slope (normalized)
        if len(volumes) >= 5:
            vol_5d = volumes[-5:]
            x = np.arange(len(vol_5d))
            # Simple linear regression slope
            if len(vol_5d) > 1 and np.std(x) > 0:
                slope = np.corrcoef(x, vol_5d)[0, 1] * (np.std(vol_5d) / np.std(x))
                # Normalize by average volume
                if vol_20d_avg > 0:
                    ctx.volume_trend = slope / vol_20d_avg * 100  # % per day

    # --- TIER 2: Intraday Volatility Features ---
    if 'high' in price_history.columns and 'low' in price_history.columns:
        if len(price_history) >= 1:
            current_high = price_history.iloc[-1]['high']
            current_low = price_history.iloc[-1]['low']
            current_close = price_history.iloc[-1]['close']

            # Intraday range
            if current_close > 0:
                ctx.intraday_range = (current_high - current_low) / current_close * 100

        # Gap size (open vs previous close)
        if len(price_history) >= 2 and 'open' in price_history.columns:
            prev_close = price_history.iloc[-2]['close']
            current_open = price_history.iloc[-1].get('open', current_close)
            if prev_close > 0:
                ctx.gap_size = (current_open - prev_close) / prev_close * 100

    # --- TIER 2: Earnings Context ---
    # Note: Earnings data would come from iv_data parameter
    # For now, we'll leave this as optional/future enhancement
    # The model will use default values (days_to_earnings=100, earnings_within_window=False)

    # --- Position Features ---
    if len(closes) >= 20:
        sma20 = closes[-20:].mean()
        ctx.position_vs_sma20 = (current_price - sma20) / sma20 * 100

    if len(closes) >= 50:
        sma50 = closes[-50:].mean()
        ctx.position_vs_sma50 = (current_price - sma50) / sma50 * 100

    if len(closes) >= 10:
        last_10 = closes[-10:]
        min_10 = last_10.min()
        max_10 = last_10.max()
        if max_10 > min_10:
            ctx.position_in_10d_range = (current_price - min_10) / (max_10 - min_10) * 100
        else:
            ctx.position_in_10d_range = 50.0

    if len(closes) >= 20:
        last_20 = closes[-20:]
        min_20 = last_20.min()
        max_20 = last_20.max()
        if max_20 > min_20:
            ctx.position_in_20d_range = (current_price - min_20) / (max_20 - min_20) * 100
        else:
            ctx.position_in_20d_range = 50.0

        ctx.distance_from_high_20d = (max_20 - current_price) / max_20 * 100
        ctx.distance_from_low_20d = (current_price - min_20) / min_20 * 100

    # --- Momentum Features ---
    if len(closes) >= 2:
        ctx.return_1d = (closes[-1] - closes[-2]) / closes[-2] * 100

    if len(closes) >= 5:
        ctx.return_5d = (closes[-1] - closes[-5]) / closes[-5] * 100

    if len(closes) >= 10:
        ctx.return_10d = (closes[-1] - closes[-10]) / closes[-10] * 100

    if len(closes) >= 20:
        ctx.return_20d = (closes[-1] - closes[-20]) / closes[-20] * 100

    # Consecutive up/down days
    if len(closes) >= 10:
        daily_changes = np.diff(closes[-10:])
        consecutive = 0
        for i in range(len(daily_changes) - 1, -1, -1):
            if i == len(daily_changes) - 1:
                consecutive = 1 if daily_changes[i] > 0 else -1
            elif (daily_changes[i] > 0 and consecutive > 0) or (daily_changes[i] < 0 and consecutive < 0):
                consecutive += 1 if daily_changes[i] > 0 else -1
            else:
                break
        ctx.consecutive_days = consecutive

    # Trend strength (simplified ADX using range)
    if len(closes) >= 20:
        highs = price_history['high'].values[-20:]
        lows = price_history['low'].values[-20:]

        # True range
        true_ranges = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[-20:-1]),
                np.abs(lows[1:] - closes[-20:-1])
            )
        )
        atr = true_ranges.mean()

        # Directional movement
        up_moves = highs[1:] - highs[:-1]
        down_moves = lows[:-1] - lows[1:]

        plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0).mean()
        minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0).mean()

        plus_di = (plus_dm / atr * 100) if atr > 0 else 0
        minus_di = (minus_dm / atr * 100) if atr > 0 else 0

        # ADX approximation
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
        ctx.trend_strength = min(dx, 100.0)

    # --- Calendar Features ---
    if isinstance(current_date, date):
        dt = datetime.combine(current_date, datetime.min.time())
    else:
        dt = current_date

    ctx.day_of_week = dt.weekday()  # 0=Mon, 4=Fri
    ctx.month = dt.month

    # OPEX week (3rd Friday of month = days 15-21)
    ctx.is_opex_week = (15 <= dt.day <= 21)

    # Days to month end (approximate as trading days)
    import calendar
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    days_left = last_day - dt.day
    ctx.days_to_month_end = max(0, int(days_left * 5 / 7))  # rough conversion to trading days

    # --- Derived Features ---
    ctx.is_overbought = ctx.position_vs_sma20 > 3.0
    ctx.is_oversold = ctx.position_vs_sma20 < -3.0
    ctx.is_trending = ctx.trend_strength > 25.0

    return ctx


def compute_historical_contexts(
    ticker: str,
    all_dates: List[str],
    price_data_by_date: Dict[str, pd.DataFrame],
    vix_data_by_date: Optional[Dict[str, pd.DataFrame]] = None,
    lookback_days: int = 60,
) -> List[MarketContext]:
    """Compute market context for each historical date.

    Args:
        ticker: Ticker symbol
        all_dates: List of dates (sorted ascending) as strings 'YYYY-MM-DD'
        price_data_by_date: Dict mapping date -> DataFrame for that date with OHLCV
        vix_data_by_date: Optional dict mapping date -> VIX data
        lookback_days: Days of history needed for feature computation

    Returns:
        List of MarketContext objects, one per date (parallel to all_dates)
    """
    contexts = []

    for i, current_date_str in enumerate(all_dates):
        # Build price history DataFrame from lookback_days before current date
        start_idx = max(0, i - lookback_days)
        history_dates = all_dates[start_idx:i+1]

        history_rows = []
        for d in history_dates:
            if d in price_data_by_date:
                df = price_data_by_date[d]
                if not df.empty:
                    row = {
                        'date': d,
                        'close': df.iloc[-1]['close'],
                        'high': df['high'].max() if 'high' in df.columns else df.iloc[-1]['close'],
                        'low': df['low'].min() if 'low' in df.columns else df.iloc[-1]['close'],
                        'volume': df['volume'].sum() if 'volume' in df.columns else 0,
                    }
                    history_rows.append(row)

        if not history_rows:
            contexts.append(MarketContext())  # defaults
            continue

        price_history = pd.DataFrame(history_rows)
        current_price = price_history.iloc[-1]['close']
        current_date = datetime.strptime(current_date_str, '%Y-%m-%d').date()

        # Build VIX history if available
        vix_history = None
        if vix_data_by_date:
            vix_rows = []
            for d in history_dates:
                if d in vix_data_by_date:
                    vdf = vix_data_by_date[d]
                    if not vdf.empty:
                        vix_rows.append({
                            'date': d,
                            'close': vdf.iloc[-1]['close'],
                        })
            if vix_rows:
                vix_history = pd.DataFrame(vix_rows)

        ctx = compute_market_context(
            ticker=ticker,
            current_price=current_price,
            current_date=current_date,
            price_history=price_history,
            vix_history=vix_history,
        )
        contexts.append(ctx)

    return contexts


def compute_feature_similarity(ctx1: MarketContext, ctx2: MarketContext) -> float:
    """Compute similarity score (0-1) between two market contexts.

    Higher score = more similar conditions = more relevant historical sample.
    """
    # Volatility similarity (most important)
    vix_diff = abs((ctx1.vix or 15) - (ctx2.vix or 15))
    vol_sim = 1.0 / (1.0 + vix_diff / 5.0)  # decay over VIX=5 difference

    # Position similarity
    pos_diff = abs(ctx1.position_vs_sma20 - ctx2.position_vs_sma20)
    pos_sim = 1.0 / (1.0 + pos_diff / 2.0)  # decay over 2% difference

    # Momentum similarity
    mom_diff = abs(ctx1.return_5d - ctx2.return_5d)
    mom_sim = 1.0 / (1.0 + mom_diff / 3.0)  # decay over 3% difference

    # Calendar similarity (day of week matters less for multi-day)
    cal_sim = 1.0 if ctx1.day_of_week == ctx2.day_of_week else 0.8

    # Weighted combination
    similarity = (
        0.40 * vol_sim +
        0.30 * pos_sim +
        0.20 * mom_sim +
        0.10 * cal_sim
    )

    return similarity
