"""
Core prediction pipeline: training, per-model prediction, and unified combination.
"""

import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import (
    UnifiedBand,
    UnifiedPrediction,
    STAT_FEATURE_CONFIG,
    USE_LGBM_PREDICTOR,
    LGBM_N_ESTIMATORS,
    LGBM_LEARNING_RATE,
    LGBM_MAX_DEPTH,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_BAND_WIDTH_SCALE,
    ENABLE_DYNAMIC_VOL_SCALING,
    VOL_LOOKBACK_DAYS,
    VOL_BASELINE_DAYS,
)
from .bands import map_statistical_to_bands, map_percentile_to_bands, combine_bands
from .features import detect_reversal_strength

from scripts.percentile_range_backtest import (
    collect_all_data,
    vol_scale_moves,
    HOURS_TO_CLOSE,
)
from scripts.strategy_utils.close_predictor import (
    PredictionContext,
    ClosePrediction,
    StatisticalClosePredictor,
)
from scripts.csv_prediction_backtest import (
    get_available_dates,
    build_training_data,
    DayContext,
)


def train_both_models(
    ticker: str,
    lookback: int = 250,
    test_date: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[StatisticalClosePredictor], List[str]]:
    """Load CSV data, train the statistical predictor, and collect percentile data.

    Returns:
        (percentile_df, statistical_predictor, all_dates)
    """
    needed = lookback + 20
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < lookback + 5:
        print(f"Not enough data. Have {len(all_dates)} dates, need at least {lookback + 5}.")
        return None, None, all_dates

    # Percentile model data
    pct_df = collect_all_data(ticker, all_dates)

    # Statistical model
    if test_date is None:
        test_date = all_dates[-1]
    stat_predictor = _train_statistical(ticker, test_date, lookback)

    return pct_df, stat_predictor, all_dates


def _train_statistical(
    ticker: str,
    test_date: str,
    lookback_days: int,
    use_lgbm: bool = None,
) -> Optional[StatisticalClosePredictor]:
    """Train StatisticalClosePredictor or LGBMClosePredictor.

    Args:
        ticker: Ticker symbol
        test_date: Test date
        lookback_days: Number of days for training
        use_lgbm: If True, use LGBMClosePredictor; if None, use USE_LGBM_PREDICTOR config

    Returns:
        Predictor instance (LGBMClosePredictor or StatisticalClosePredictor)
    """
    train_df = build_training_data(ticker, test_date, lookback_days)
    if train_df.empty or len(train_df) < 50:
        return None

    # Determine whether to use LightGBM
    if use_lgbm is None:
        use_lgbm = USE_LGBM_PREDICTOR

    if use_lgbm:
        try:
            from scripts.strategy_utils.close_predictor import LGBMClosePredictor

            predictor = LGBMClosePredictor(
                n_estimators=LGBM_N_ESTIMATORS,
                learning_rate=LGBM_LEARNING_RATE,
                max_depth=LGBM_MAX_DEPTH,
                min_child_samples=LGBM_MIN_CHILD_SAMPLES,
                band_width_scale=LGBM_BAND_WIDTH_SCALE,
                use_fallback=True,  # Graceful fallback
            )
            predictor.fit(train_df)
            return predictor
        except Exception as e:
            print(f"LGBMClosePredictor failed: {e}, falling back to Statistical")
            # Fall through to statistical predictor

    # Original statistical predictor
    predictor = StatisticalClosePredictor(
        min_samples=5,
        **STAT_FEATURE_CONFIG,
    )
    predictor.fit(train_df)
    return predictor


def compute_static_percentile_prediction(
    pct_df: pd.DataFrame,
    prev_close: float,
    train_dates: set,
) -> Optional[Dict[str, UnifiedBand]]:
    """Compute static close-to-close percentile bands (no time/vol/direction filtering).

    Uses all historical prev_close-to-day_close returns and applies them
    against prev_close. This produces bands that do not change throughout the day.
    """
    # Use one record per date (deduplicate time slots) — take the first slot per date
    daily = pct_df[pct_df['date'].isin(train_dates)].drop_duplicates(subset='date')
    if len(daily) < 10:
        return None

    # Compute prev_close → day_close returns
    moves = ((daily['day_close'] - daily['prev_close']) / daily['prev_close'] * 100).values

    return map_percentile_to_bands(moves, prev_close)


def compute_percentile_prediction(
    pct_df: pd.DataFrame,
    time_label: str,
    above: bool,
    current_price: float,
    current_vol: Optional[float],
    train_dates: set,
    vol_scale: bool = True,
    reversal_blend: float = 0.0,
    intraday_vol_factor: float = 1.0,
) -> Optional[Dict[str, UnifiedBand]]:
    """Compute percentile-model bands for a single time slot.

    Args:
        reversal_blend: Weight in [0, 0.5] for blending opposite-condition data.
            When > 0, fetches opposite-condition training data and blends it in.
        intraday_vol_factor: Scaling factor for intraday vol adaptation.
            Values > 1 widen bands, < 1 tighten them.
    """
    train_slot = pct_df[
        (pct_df['time'] == time_label) &
        (pct_df['above'] == above) &
        (pct_df['date'].isin(train_dates))
    ]
    if len(train_slot) < 10:
        # Fall back: ignore above/below filter
        train_slot = pct_df[
            (pct_df['time'] == time_label) &
            (pct_df['date'].isin(train_dates))
        ]
    if len(train_slot) < 10:
        return None

    if vol_scale:
        moves = vol_scale_moves(train_slot, current_vol)
    else:
        moves = train_slot['close_move_pct'].values

    # Reversal blending: mix in opposite-condition training data
    if reversal_blend > 0:
        opp_slot = pct_df[
            (pct_df['time'] == time_label) &
            (pct_df['above'] == (not above)) &
            (pct_df['date'].isin(train_dates))
        ]
        if len(opp_slot) >= 5:
            if vol_scale:
                opp_moves = vol_scale_moves(opp_slot, current_vol)
            else:
                opp_moves = opp_slot['close_move_pct'].values

            n_same = len(moves)
            n_opp = int(n_same * reversal_blend / (1 - reversal_blend))
            n_opp = min(n_opp, len(opp_moves))

            if n_opp > 0:
                sampled_opp = np.array(random.sample(list(opp_moves), n_opp))
                moves = np.concatenate([moves, sampled_opp])

    # Remove extreme outliers (beyond 3 standard deviations)
    # This prevents rare extreme events from distorting the bands
    std_dev = np.std(moves)
    mean_move = np.mean(moves)
    outlier_threshold = 3.0
    mask = np.abs(moves - mean_move) <= (outlier_threshold * std_dev)
    moves_clean = moves[mask]

    # Only use cleaned data if we have enough samples left
    if len(moves_clean) >= max(10, len(moves) * 0.8):
        moves = moves_clean

    # Intraday vol adaptation: scale moves around their mean
    # NOTE: This is typically disabled (factor=1.0) to match validated backtest results
    if intraday_vol_factor != 1.0:
        mean_move_final = np.mean(moves)
        moves = mean_move_final + (moves - mean_move_final) * intraday_vol_factor

    return map_percentile_to_bands(moves, current_price)


def compute_statistical_prediction(
    predictor: StatisticalClosePredictor,
    ticker: str,
    current_price: float,
    current_time: datetime,
    day_ctx: DayContext,
    day_high: float,
    day_low: float,
) -> Tuple[Optional[Dict[str, UnifiedBand]], Optional[ClosePrediction]]:
    """Compute statistical-model bands."""
    if predictor is None:
        return None, None

    db_ticker = f"I:{ticker}" if not ticker.startswith("I:") else ticker

    # Compute volatility if dynamic scaling is enabled
    realized_vol = None
    historical_avg_vol = None
    if ENABLE_DYNAMIC_VOL_SCALING:
        try:
            from scripts.csv_prediction_backtest import (
                get_trailing_realized_vol,
                get_historical_avg_vol,
            )

            date_str = current_time.strftime('%Y-%m-%d')
            realized_vol = get_trailing_realized_vol(ticker, date_str, VOL_LOOKBACK_DAYS)
            historical_avg_vol = get_historical_avg_vol(ticker, date_str, VOL_BASELINE_DAYS)
        except Exception as e:
            # Silently continue if volatility computation fails
            pass

    context = PredictionContext(
        ticker=db_ticker,
        current_price=current_price,
        prev_close=day_ctx.prev_close,
        day_open=day_ctx.day_open,
        current_time=current_time,
        vix1d=day_ctx.vix1d if day_ctx.vix1d else 15.0,
        day_high=day_high,
        day_low=day_low,
        prev_day_close=day_ctx.prev_day_close,
        prev_vix1d=day_ctx.prev_vix1d,
        prev_day_high=day_ctx.prev_day_high,
        prev_day_low=day_ctx.prev_day_low,
        close_5days_ago=day_ctx.close_5days_ago,
        first_hour_high=day_ctx.first_hour_high,
        first_hour_low=day_ctx.first_hour_low,
        opening_range_high=day_ctx.opening_range_high,
        opening_range_low=day_ctx.opening_range_low,
        price_at_945=day_ctx.price_at_945,
        ma5=day_ctx.ma5,
        ma10=day_ctx.ma10,
        ma20=day_ctx.ma20,
        ma50=day_ctx.ma50,
        realized_vol=realized_vol,
        historical_avg_vol=historical_avg_vol,
    )

    try:
        prediction = predictor.predict(context)
    except Exception:
        return None, None

    bands = map_statistical_to_bands(prediction, current_price)
    return bands, prediction


def make_unified_prediction(
    pct_df: pd.DataFrame,
    predictor: Optional[StatisticalClosePredictor],
    ticker: str,
    current_price: float,
    prev_close: float,
    current_time: datetime,
    time_label: str,
    day_ctx: DayContext,
    day_high: float,
    day_low: float,
    train_dates: set,
    current_vol: Optional[float] = None,
    vol_scale: bool = True,
    data_source: str = "csv",
    intraday_vol_factor: float = 1.0,
    use_gap_adjustment: bool = True,
) -> Optional[UnifiedPrediction]:
    """Produce a unified prediction combining both models."""
    above = current_price >= prev_close
    hours_left = HOURS_TO_CLOSE.get(time_label, 0)

    # Compute reversal blend
    day_open = day_ctx.day_open if day_ctx.day_open else current_price
    reversal_blend = detect_reversal_strength(
        current_price, prev_close, day_open, day_high, day_low,
    )

    # Percentile model — static close-to-close (does not change throughout the day)
    pct_bands = compute_static_percentile_prediction(pct_df, prev_close, train_dates)

    # Time-aware percentile model (used for combined bands only)
    time_aware_pct_bands = compute_percentile_prediction(
        pct_df, time_label, above, current_price, current_vol, train_dates, vol_scale,
        reversal_blend=reversal_blend,
        intraday_vol_factor=intraday_vol_factor,
    )

    # Statistical model
    stat_bands, stat_pred = compute_statistical_prediction(
        predictor, ticker, current_price, current_time,
        day_ctx, day_high, day_low,
    )

    # Need at least one model
    if pct_bands is None and time_aware_pct_bands is None and stat_bands is None:
        return None

    if pct_bands is None:
        pct_bands = {}
    if stat_bands is None:
        stat_bands = {}

    # Combined uses time-aware percentile (more adaptive) blended with statistical
    combined_pct = time_aware_pct_bands if time_aware_pct_bands else pct_bands
    combined = combine_bands(combined_pct, stat_bands, current_price)

    # Apply opening gap adjustment if enabled and early in trading day
    gap_adjustment_applied = False
    if use_gap_adjustment and hours_left >= 4.0:  # Only in first ~2.5 hours
        from scripts.close_predictor.opening_gap_model import (
            adjust_unified_bands_for_gap,
            get_gap_summary,
        )

        # Convert current_time to hour (9.5 = 9:30 AM, etc.)
        hour = current_time.hour + current_time.minute / 60.0

        # Adjust statistical and combined bands (not static percentile — it stays fixed)
        stat_bands = adjust_unified_bands_for_gap(stat_bands, current_price, prev_close, hour)
        combined = adjust_unified_bands_for_gap(combined, current_price, prev_close, hour)

        # Check if adjustment was actually applied (gap is significant)
        from scripts.close_predictor.opening_gap_model import detect_opening_gap
        gap_analysis = detect_opening_gap(current_price, prev_close)
        if gap_analysis.is_significant:
            gap_adjustment_applied = True
            print(f"  Opening gap adjustment: {get_gap_summary(current_price, prev_close, hour)}")

    # Apply late-day volatility buffer (2:30 PM onwards)
    late_day_adjustment_applied = False
    if hours_left <= 1.5:  # 1.5 hours or less to close (2:30 PM onwards)
        from scripts.close_predictor.late_day_buffer import (
            adjust_bands_for_late_day,
            get_late_day_summary,
            get_late_day_multiplier,
        )

        # Convert current_time to hour
        hour = current_time.hour + current_time.minute / 60.0

        # Get multiplier to check if adjustment will be applied
        multiplier = get_late_day_multiplier(hour)

        if multiplier > 1.0:
            # Adjust statistical and combined bands (not static percentile — it stays fixed)
            stat_bands = adjust_bands_for_late_day(stat_bands, hour)
            combined = adjust_bands_for_late_day(combined, hour)

            late_day_adjustment_applied = True
            print(f"  Late-day buffer: {get_late_day_summary(hour)}")

    confidence = stat_pred.confidence.value if stat_pred else None
    risk_level = stat_pred.recommended_risk_level if stat_pred else None
    vix1d = day_ctx.vix1d
    sample_size = stat_pred.sample_size if stat_pred else None

    # Simplified 0DTE directional analysis using intraday momentum
    directional = None
    try:
        daily = pct_df[pct_df['date'].isin(train_dates)].drop_duplicates(subset='date')
        if len(daily) >= 20:
            moves = ((daily['day_close'] - daily['prev_close']) / daily['prev_close'] * 100).values
            from .directional_analysis import (
                MomentumState, DirectionalProbability,
                compute_asymmetric_bands,
            )
            # For 0DTE, use above_prev as simplified momentum signal
            up_count = int((moves > 0).sum())
            down_count = int((moves < 0).sum())
            total = len(moves)
            p_up = up_count / total if total > 0 else 0.5
            p_down = down_count / total if total > 0 else 0.5
            # Skew probability by current intraday direction
            if above:
                p_up = min(1.0, p_up * 1.1)
                p_down = 1.0 - p_up
            else:
                p_down = min(1.0, p_down * 1.1)
                p_up = 1.0 - p_down
            dir_prob = DirectionalProbability(
                p_up=round(p_up, 4), p_down=round(p_down, 4),
                up_count=up_count, down_count=down_count,
                total_samples=total,
                confidence="medium" if total >= 30 else "low",
                mean_reversion_prob=0.5,
            )
            momentum = MomentumState(
                trend_label="up" if above else "down",
                consecutive_days=0,
                return_5d=0.0,
                is_extended_streak=False,
            )
            from .directional_analysis import DirectionalAnalysis
            asym_bands = compute_asymmetric_bands(moves, current_price, dir_prob)
            directional = DirectionalAnalysis(
                momentum_state=momentum,
                direction_probability=dir_prob,
                asymmetric_bands=asym_bands,
            )
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"0DTE directional analysis skipped: {e}")

    return UnifiedPrediction(
        ticker=ticker,
        current_price=current_price,
        prev_close=prev_close,
        hours_to_close=hours_left,
        time_label=time_label,
        above_prev=above,
        percentile_bands=pct_bands,
        statistical_bands=stat_bands,
        combined_bands=combined,
        confidence=confidence,
        risk_level=risk_level,
        vix1d=vix1d,
        realized_vol=current_vol,
        stat_sample_size=sample_size,
        reversal_blend=reversal_blend,
        intraday_vol_factor=intraday_vol_factor,
        data_source=data_source,
        directional_analysis=directional,
    )
