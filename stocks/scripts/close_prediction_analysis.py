#!/usr/bin/env python3
"""
Closing Price Prediction Analysis CLI.

Predicts where the closing price will be based on:
1. Current price at the prediction hour
2. VIX1D (1-day volatility index) level
3. Overnight gap (open vs previous close)

Usage:
    # Get current prediction
    python scripts/close_prediction_analysis.py --ticker NDX

    # Backtest accuracy
    python scripts/close_prediction_analysis.py --ticker NDX --backtest --days 30

    # Analyze patterns
    python scripts/close_prediction_analysis.py --ticker NDX --analyze-patterns

    # Train and save model
    python scripts/close_prediction_analysis.py --ticker NDX --train --save-model ./models/ndx_predictor
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Project Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger

from credit_spread_utils.price_utils import (
    get_historical_price_patterns,
    get_previous_close_price,
    get_current_day_open_price,
    get_price_at_time,
    get_current_vix1d,
    get_intraday_high_low,
)
from credit_spread_utils.timezone_utils import get_timezone, normalize_timestamp

from strategy_utils.close_predictor import (
    PredictionContext,
    ClosePrediction,
    StatisticalClosePredictor,
    MLClosePredictor,
    EnsemblePredictor,
    format_prediction_report,
    VIXRegime,
    GapType,
)


def generate_synthetic_data(
    ticker: str,
    days: int = 365,
    base_price: float = 17500.0,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Generate synthetic historical data for demo/testing.

    Creates realistic-looking price patterns with:
    - Daily volatility based on VIX levels
    - Overnight gaps
    - Intraday price movements

    Args:
        ticker: Ticker symbol
        days: Number of days of data to generate
        base_price: Starting price
        logger: Optional logger

    Returns:
        DataFrame with columns matching get_historical_price_patterns output
    """
    if logger:
        logger.info(f"Generating {days} days of synthetic data for {ticker}")

    np.random.seed(42)  # For reproducibility
    records = []

    current_price = base_price
    hours = [9, 10, 11, 12, 13, 14, 15]

    for day_offset in range(days):
        # Generate VIX for the day (mean-reverting around 17)
        vix_base = 17 + np.random.randn() * 5
        vix1d = max(10, min(40, vix_base))

        # Daily volatility based on VIX
        daily_vol = (vix1d / 100) * (1 / np.sqrt(252))

        # Previous close
        prev_close = current_price

        # Overnight gap (-1% to +1%, biased toward smaller moves)
        gap_pct = np.random.randn() * 0.003  # ~0.3% std dev
        day_open = prev_close * (1 + gap_pct)

        # Intraday high/low range
        intraday_range = day_open * daily_vol * np.random.uniform(0.5, 1.5)
        day_high = day_open + intraday_range * np.random.uniform(0.3, 0.7)
        day_low = day_open - intraday_range * np.random.uniform(0.3, 0.7)

        # Day close (random walk from open)
        close_move = np.random.randn() * daily_vol
        day_close = day_open * (1 + close_move)
        day_close = max(day_low, min(day_high, day_close))

        # Generate date
        trade_date = datetime.now().date() - timedelta(days=days - day_offset)
        day_of_week = trade_date.weekday()

        # Skip weekends
        if day_of_week >= 5:
            continue

        # Generate hourly prices
        for hour in hours:
            # Price at this hour (interpolate between open and close)
            progress = (hour - 9) / 6.5  # 0 at 9:00, 1 at 15:30
            noise = np.random.randn() * daily_vol * 0.3
            hour_price = day_open + (day_close - day_open) * progress + day_open * noise

            # Clamp to day range
            hour_price = max(day_low, min(day_high, hour_price))

            records.append({
                'date': trade_date,
                'hour_et': hour,
                'hour_price': round(hour_price, 2),
                'day_open': round(day_open, 2),
                'day_close': round(day_close, 2),
                'prev_close': round(prev_close, 2),
                'vix1d': round(vix1d, 2),
                'day_of_week': day_of_week,
            })

        # Update current price for next day
        current_price = day_close

    df = pd.DataFrame(records)
    if logger:
        logger.info(f"Generated {len(df)} synthetic records")

    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Closing Price Prediction Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get current prediction for NDX
  python close_prediction_analysis.py --ticker NDX

  # Backtest prediction accuracy over last 30 days
  python close_prediction_analysis.py --ticker NDX --backtest --days 30

  # Analyze historical patterns
  python close_prediction_analysis.py --ticker NDX --analyze-patterns

  # Train model and save
  python close_prediction_analysis.py --ticker NDX --train --save-model ./models/ndx

  # Load saved model and predict
  python close_prediction_analysis.py --ticker NDX --load-model ./models/ndx
        """
    )

    # Required arguments
    parser.add_argument(
        '--ticker', '-t',
        type=str,
        required=True,
        help='Ticker symbol (e.g., NDX, SPX). Will be prefixed with I: for index lookup.'
    )

    # Mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--predict',
        action='store_true',
        default=True,
        help='Make a prediction for current time (default mode)'
    )
    mode_group.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest on historical data'
    )
    mode_group.add_argument(
        '--analyze-patterns',
        action='store_true',
        help='Analyze historical price patterns'
    )
    mode_group.add_argument(
        '--train',
        action='store_true',
        help='Train prediction model on historical data'
    )

    # Model arguments
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['statistical', 'ml', 'ensemble'],
        default='ensemble',
        help='Type of prediction model to use (default: ensemble)'
    )
    parser.add_argument(
        '--ml-type',
        type=str,
        choices=['xgboost', 'random_forest'],
        default='xgboost',
        help='ML model type for ML/ensemble predictor (default: xgboost)'
    )
    parser.add_argument(
        '--save-model',
        type=str,
        help='Path to save trained model'
    )
    parser.add_argument(
        '--load-model',
        type=str,
        help='Path to load saved model'
    )

    # Data arguments
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=365,
        help='Number of days of historical data to use (default: 365)'
    )
    parser.add_argument(
        '--backtest-days',
        type=int,
        default=30,
        help='Number of days to backtest (default: 30)'
    )

    # Prediction arguments
    parser.add_argument(
        '--hour',
        type=int,
        choices=range(9, 16),
        help='Hour (ET) to make prediction for. Default: current hour.'
    )
    parser.add_argument(
        '--price',
        type=float,
        help='Current price to use (for testing). Default: fetch from DB.'
    )

    # Output arguments
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['text', 'json', 'csv'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file path (default: stdout)'
    )

    # Database arguments
    parser.add_argument(
        '--db-config',
        type=str,
        default='questdb://admin:quest@localhost:8812/qdb',
        help='Database connection string'
    )

    # Demo mode
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode with synthetic data (no database required)'
    )

    # Logging arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Normalize ticker
    ticker = args.ticker.upper()
    if not ticker.startswith('I:'):
        ticker = f'I:{ticker}'
    args.ticker = ticker

    return args


async def get_current_context(
    db: StockQuestDB,
    ticker: str,
    hour: Optional[int] = None,
    override_price: Optional[float] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[PredictionContext]:
    """
    Build prediction context from current market data.

    Args:
        db: Database connection
        ticker: Ticker symbol
        hour: Override hour (ET). Default: current hour.
        override_price: Override current price (for testing)
        logger: Optional logger

    Returns:
        PredictionContext or None if data unavailable
    """
    et_tz = get_timezone("America/New_York")
    now_et = datetime.now(et_tz)

    if hour is not None:
        # Override hour for testing
        now_et = now_et.replace(hour=hour, minute=30, second=0, microsecond=0)

    # Get previous close
    prev_close_result = await get_previous_close_price(db, ticker, now_et, logger)
    if prev_close_result is None:
        if logger:
            logger.error(f"Could not get previous close for {ticker}")
        return None
    prev_close, _ = prev_close_result

    # Get today's open
    day_open = await get_current_day_open_price(db, ticker, now_et, logger)
    if day_open is None:
        if logger:
            logger.warning(f"Could not get day open for {ticker}, using prev_close")
        day_open = prev_close

    # Get current price
    if override_price is not None:
        current_price = override_price
    else:
        current_price = await get_price_at_time(db, ticker, now_et, logger)
        if current_price is None:
            if logger:
                logger.warning(f"Could not get current price, using day open")
            current_price = day_open

    # Get VIX1D
    vix1d = await get_current_vix1d(db, logger)

    # Get intraday high/low
    high_low = await get_intraday_high_low(db, ticker, now_et, logger)
    day_high, day_low = high_low if high_low else (None, None)

    return PredictionContext(
        ticker=ticker,
        current_price=current_price,
        prev_close=prev_close,
        day_open=day_open,
        current_time=now_et,
        vix1d=vix1d,
        day_high=day_high,
        day_low=day_low,
        day_of_week=now_et.weekday(),
    )


async def run_prediction(
    args: argparse.Namespace,
    logger: logging.Logger
) -> Optional[ClosePrediction]:
    """Run prediction for current time."""
    logger.info(f"Making prediction for {args.ticker}")

    # Demo mode - use synthetic data
    if args.demo:
        return await run_prediction_demo(args, logger)

    # Connect to database
    db = StockQuestDB(db_config=args.db_config)
    await db._init_db()

    try:
        # Get current context
        context = await get_current_context(
            db, args.ticker, args.hour, args.price, logger
        )
        if context is None:
            logger.error("Could not build prediction context")
            return None

        # Load or create predictor
        if args.load_model:
            logger.info(f"Loading model from {args.load_model}")
            predictor = EnsemblePredictor(logger=logger)
            predictor.load(args.load_model)
        else:
            # Train on historical data
            logger.info(f"Fetching {args.days} days of historical data...")
            df = await get_historical_price_patterns(
                db, args.ticker, args.days, logger=logger
            )

            if df.empty:
                logger.error("No historical data available for training")
                return None

            logger.info(f"Training on {len(df)} records...")

            if args.model_type == 'statistical':
                predictor = StatisticalClosePredictor(logger=logger)
                predictor.fit(df)
            elif args.model_type == 'ml':
                predictor = MLClosePredictor(
                    model_type=args.ml_type, logger=logger
                )
                predictor.fit(df)
            else:  # ensemble
                predictor = EnsemblePredictor(logger=logger)
                predictor.fit(df, fit_ml=True, ml_model_type=args.ml_type)

            if args.save_model:
                logger.info(f"Saving model to {args.save_model}")
                predictor.save(args.save_model)

        # Make prediction
        prediction = predictor.predict(context)
        return prediction

    finally:
        await db.close()


async def run_prediction_demo(
    args: argparse.Namespace,
    logger: logging.Logger
) -> Optional[ClosePrediction]:
    """Run prediction in demo mode with synthetic data."""
    logger.info("Running in DEMO mode with synthetic data")

    # Generate synthetic historical data
    df = generate_synthetic_data(args.ticker, args.days, logger=logger)

    if df.empty:
        logger.error("Failed to generate synthetic data")
        return None

    # Use most recent day's data for current context
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date].iloc[-1]

    # Build context
    et_tz = get_timezone("America/New_York")
    current_hour = args.hour if args.hour else 11  # Default to 11 AM
    current_time = datetime.now(et_tz).replace(hour=current_hour, minute=30)

    context = PredictionContext(
        ticker=args.ticker,
        current_price=args.price if args.price else float(latest_data['hour_price']),
        prev_close=float(latest_data['prev_close']),
        day_open=float(latest_data['day_open']),
        current_time=current_time,
        vix1d=float(latest_data['vix1d']),
        day_of_week=current_time.weekday(),
    )

    logger.info(f"Training on {len(df)} synthetic records...")

    # Train predictor
    if args.model_type == 'statistical':
        predictor = StatisticalClosePredictor(logger=logger)
        predictor.fit(df)
    elif args.model_type == 'ml':
        predictor = MLClosePredictor(model_type=args.ml_type, logger=logger)
        predictor.fit(df)
    else:  # ensemble
        predictor = EnsemblePredictor(logger=logger)
        predictor.fit(df, fit_ml=True, ml_model_type=args.ml_type)

    if args.save_model:
        logger.info(f"Saving model to {args.save_model}")
        predictor.save(args.save_model)

    # Make prediction
    prediction = predictor.predict(context)
    return prediction


async def run_backtest(
    args: argparse.Namespace,
    logger: logging.Logger
) -> pd.DataFrame:
    """Run backtest on historical data."""
    logger.info(f"Running backtest for {args.ticker} over {args.backtest_days} days")

    # Get historical data
    if args.demo:
        logger.info("Running in DEMO mode with synthetic data")
        df = generate_synthetic_data(args.ticker, args.days, logger=logger)
    else:
        # Connect to database
        db = StockQuestDB(db_config=args.db_config)
        await db._init_db()

        try:
            logger.info(f"Fetching {args.days} days of historical data...")
            df = await get_historical_price_patterns(
                db, args.ticker, args.days, logger=logger
            )
        finally:
            await db.close()

    if df.empty:
        logger.error("No historical data available")
        return pd.DataFrame()

    # Split into training and test sets
    # Sort by date
    df = df.sort_values(['date', 'hour_et']).reset_index(drop=True)

    # Get unique dates
    unique_dates = df['date'].unique()
    test_dates = unique_dates[-args.backtest_days:]
    train_dates = unique_dates[:-args.backtest_days]

    if len(train_dates) < 50:
        logger.error(f"Insufficient training data: {len(train_dates)} days")
        return pd.DataFrame()

    train_df = df[df['date'].isin(train_dates)]
    test_df = df[df['date'].isin(test_dates)]

    logger.info(f"Training on {len(train_df)} records ({len(train_dates)} days)")
    logger.info(f"Testing on {len(test_df)} records ({len(test_dates)} days)")

    # Train predictor
    if args.model_type == 'statistical':
        predictor = StatisticalClosePredictor(logger=logger)
        predictor.fit(train_df)
    elif args.model_type == 'ml':
        predictor = MLClosePredictor(model_type=args.ml_type, logger=logger)
        predictor.fit(train_df)
    else:  # ensemble
        predictor = EnsemblePredictor(logger=logger)
        predictor.fit(train_df, fit_ml=True, ml_model_type=args.ml_type)

    # Run predictions on test data
    results = []
    for _, row in test_df.iterrows():
        try:
            # Build context from historical data
            et_tz = get_timezone("America/New_York")
            test_time = datetime.combine(
                row['date'],
                datetime.min.time()
            ).replace(hour=int(row['hour_et']), minute=30, tzinfo=et_tz)

            context = PredictionContext(
                ticker=args.ticker,
                current_price=float(row['hour_price']),
                prev_close=float(row['prev_close']),
                day_open=float(row['day_open']),
                current_time=test_time,
                vix1d=row.get('vix1d'),
                day_of_week=int(row.get('day_of_week', test_time.weekday())),
            )

            prediction = predictor.predict(context)

            # Calculate actual move
            actual_close = float(row['day_close'])
            actual_move_pct = (actual_close - context.current_price) / context.current_price * 100

            # Check if actual was within predicted range
            in_range = (
                prediction.predicted_move_low_pct <= actual_move_pct <= prediction.predicted_move_high_pct
            )

            results.append({
                'date': row['date'],
                'hour_et': row['hour_et'],
                'current_price': context.current_price,
                'predicted_low': prediction.predicted_close_low,
                'predicted_mid': prediction.predicted_close_mid,
                'predicted_high': prediction.predicted_close_high,
                'actual_close': actual_close,
                'predicted_move_low_pct': prediction.predicted_move_low_pct,
                'predicted_move_mid_pct': prediction.predicted_move_mid_pct,
                'predicted_move_high_pct': prediction.predicted_move_high_pct,
                'actual_move_pct': actual_move_pct,
                'in_range': in_range,
                'confidence': prediction.confidence.value,
                'confidence_score': prediction.confidence_score,
                'risk_level': prediction.recommended_risk_level,
                'vix_regime': prediction.vix_regime,
                'sample_size': prediction.sample_size,
            })
        except Exception as e:
            logger.debug(f"Skipping row due to error: {e}")
            continue

    results_df = pd.DataFrame(results)

    if results_df.empty:
        logger.error("No backtest results generated")
        return pd.DataFrame()

    # Calculate summary statistics
    total_predictions = len(results_df)
    in_range_count = results_df['in_range'].sum()
    accuracy = in_range_count / total_predictions * 100

    mae = np.mean(np.abs(results_df['actual_move_pct'] - results_df['predicted_move_mid_pct']))
    rmse = np.sqrt(np.mean((results_df['actual_move_pct'] - results_df['predicted_move_mid_pct']) ** 2))

    # Accuracy by hour
    accuracy_by_hour = results_df.groupby('hour_et')['in_range'].mean() * 100

    # Accuracy by VIX regime
    accuracy_by_vix = results_df.groupby('vix_regime')['in_range'].mean() * 100

    print("\n" + "=" * 64)
    print(f" BACKTEST RESULTS - {args.ticker}")
    print("=" * 64)
    print(f"\nTotal Predictions: {total_predictions}")
    print(f"In-Range Accuracy: {accuracy:.1f}% ({in_range_count}/{total_predictions})")
    print(f"Mean Absolute Error: {mae:.3f}%")
    print(f"RMSE: {rmse:.3f}%")

    print("\nAccuracy by Hour (ET):")
    for hour, acc in accuracy_by_hour.items():
        print(f"  {int(hour):02d}:00 - {acc:.1f}%")

    print("\nAccuracy by VIX Regime:")
    for regime, acc in accuracy_by_vix.items():
        print(f"  {regime}: {acc:.1f}%")

    print("=" * 64)

    return results_df


async def run_pattern_analysis(
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """Analyze historical price patterns."""
    logger.info(f"Analyzing patterns for {args.ticker}")

    # Get historical data
    if args.demo:
        logger.info("Running in DEMO mode with synthetic data")
        df = generate_synthetic_data(args.ticker, args.days, logger=logger)
    else:
        # Connect to database
        db = StockQuestDB(db_config=args.db_config)
        await db._init_db()

        try:
            df = await get_historical_price_patterns(
                db, args.ticker, args.days, logger=logger
            )
        finally:
            await db.close()

    if df.empty:
        logger.error("No historical data available")
        return

    # Fit statistical predictor and get bucket stats
    predictor = StatisticalClosePredictor(logger=logger)
    predictor.fit(df)

    bucket_stats = predictor.get_bucket_stats()

    print("\n" + "=" * 80)
    print(f" PATTERN ANALYSIS - {args.ticker}")
    print("=" * 80)

    print(f"\nTotal Records: {len(df)}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Valid Buckets: {len(bucket_stats)}")

    # Summary by hour
    print("\n--- BY HOUR ---")
    hour_summary = df.groupby('hour_et').apply(
        lambda x: pd.Series({
            'count': len(x),
            'mean_move': ((x['day_close'] - x['hour_price']) / x['hour_price']).mean() * 100,
            'std_move': ((x['day_close'] - x['hour_price']) / x['hour_price']).std() * 100,
        })
    )
    print(hour_summary.to_string())

    # Summary by VIX regime
    print("\n--- BY VIX REGIME ---")
    df['vix_regime'] = df['vix1d'].apply(
        lambda x: 'LOW' if x and x < 15 else ('MEDIUM' if x and x < 20 else ('HIGH' if x and x < 30 else 'EXTREME'))
    )
    vix_summary = df.groupby('vix_regime').apply(
        lambda x: pd.Series({
            'count': len(x),
            'mean_move': ((x['day_close'] - x['hour_price']) / x['hour_price']).mean() * 100,
            'std_move': ((x['day_close'] - x['hour_price']) / x['hour_price']).std() * 100,
        })
    )
    print(vix_summary.to_string())

    # Summary by gap type
    print("\n--- BY OVERNIGHT GAP ---")
    df['gap_pct'] = (df['day_open'] - df['prev_close']) / df['prev_close'] * 100
    df['gap_type'] = df['gap_pct'].apply(
        lambda x: 'GAP_DOWN' if x < -0.5 else ('GAP_UP' if x > 0.5 else 'FLAT')
    )
    gap_summary = df.groupby('gap_type').apply(
        lambda x: pd.Series({
            'count': len(x),
            'mean_move': ((x['day_close'] - x['hour_price']) / x['hour_price']).mean() * 100,
            'std_move': ((x['day_close'] - x['hour_price']) / x['hour_price']).std() * 100,
        })
    )
    print(gap_summary.to_string())

    # Top buckets by sample count
    print("\n--- TOP BUCKETS BY SAMPLE COUNT ---")
    top_buckets = bucket_stats.nlargest(15, 'sample_count')
    print(top_buckets[['bucket_key', 'sample_count', 'mean_move', 'std_move', 'p10', 'p50', 'p90']].to_string(index=False))

    print("\n" + "=" * 80)


async def run_training(
    args: argparse.Namespace,
    logger: logging.Logger
) -> None:
    """Train and optionally save prediction model."""
    logger.info(f"Training prediction model for {args.ticker}")

    if not args.save_model:
        logger.warning("No --save-model path specified. Model will only be validated, not saved.")

    # Get historical data
    if args.demo:
        logger.info("Running in DEMO mode with synthetic data")
        df = generate_synthetic_data(args.ticker, args.days, logger=logger)
    else:
        # Connect to database
        db = StockQuestDB(db_config=args.db_config)
        await db._init_db()

        try:
            logger.info(f"Fetching {args.days} days of historical data...")
            df = await get_historical_price_patterns(
                db, args.ticker, args.days, logger=logger
            )
        finally:
            await db.close()

    if df.empty:
        logger.error("No historical data available")
        return

    logger.info(f"Training on {len(df)} records...")

    # Train predictor
    if args.model_type == 'statistical':
        predictor = StatisticalClosePredictor(logger=logger)
        predictor.fit(df)
        bucket_stats = predictor.get_bucket_stats()
        print(f"\nStatistical model trained with {len(bucket_stats)} valid buckets")
    elif args.model_type == 'ml':
        predictor = MLClosePredictor(model_type=args.ml_type, logger=logger)
        predictor.fit(df)
        print(f"\nML model ({args.ml_type}) trained")
        print(f"  Validation MAE: {predictor.validation_mae:.4f}")
        print(f"  Validation RMSE: {predictor.validation_rmse:.4f}")
        if predictor.feature_importance:
            print("  Feature Importance:")
            for feat, imp in sorted(predictor.feature_importance.items(), key=lambda x: -x[1]):
                print(f"    {feat}: {imp:.4f}")
    else:  # ensemble
        predictor = EnsemblePredictor(logger=logger)
        predictor.fit(df, fit_ml=True, ml_model_type=args.ml_type)
        bucket_stats = predictor.statistical.get_bucket_stats()
        print(f"\nEnsemble model trained:")
        print(f"  Statistical: {len(bucket_stats)} valid buckets")
        if predictor.ml and predictor.ml.is_fitted:
            print(f"  ML ({args.ml_type}): MAE={predictor.ml.validation_mae:.4f}, RMSE={predictor.ml.validation_rmse:.4f}")

    # Save model
    if args.save_model:
        logger.info(f"Saving model to {args.save_model}")
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        predictor.save(args.save_model)
        print(f"\nModel saved to: {args.save_model}")


def output_prediction(
    prediction: ClosePrediction,
    output_format: str,
    output_file: Optional[str]
) -> None:
    """Output prediction in specified format."""
    if output_format == 'text':
        output = format_prediction_report(prediction)
    elif output_format == 'json':
        import json
        output = json.dumps(prediction.to_dict(), indent=2)
    elif output_format == 'csv':
        df = pd.DataFrame([prediction.to_dict()])
        output = df.to_csv(index=False)
    else:
        output = str(prediction)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"Output written to: {output_file}")
    else:
        print(output)


async def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")
    logger = get_logger(__name__, level=log_level)

    try:
        if args.backtest:
            results_df = await run_backtest(args, logger)
            if not results_df.empty and args.output_file:
                results_df.to_csv(args.output_file, index=False)
                print(f"\nResults saved to: {args.output_file}")

        elif args.analyze_patterns:
            await run_pattern_analysis(args, logger)

        elif args.train:
            await run_training(args, logger)

        else:  # predict
            prediction = await run_prediction(args, logger)
            if prediction:
                output_prediction(prediction, args.output_format, args.output_file)
            else:
                logger.error("Failed to generate prediction")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
