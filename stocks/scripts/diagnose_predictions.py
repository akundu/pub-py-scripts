#!/usr/bin/env python3
"""Diagnostic: show per-hour predictions with feature breakdown for NDX."""
import sys
from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.csv_prediction_backtest import (
    get_available_dates, load_csv_data, build_training_data,
    get_day_open, get_day_close, get_first_hour_range, get_opening_range,
    get_price_at_time, get_vix1d_at_time, get_historical_context, DayContext,
)
from scripts.close_predictor.models import STAT_FEATURE_CONFIG
from scripts.strategy_utils.close_predictor import (
    PredictionContext, StatisticalClosePredictor, BucketFeatures,
    classify_vix_regime, classify_gap, classify_intraday_move,
    classify_prior_day_move, classify_intraday_range, classify_vix_change,
    classify_prior_close_position, classify_momentum_5day,
    classify_first_hour_range, classify_opening_drive, classify_gap_fill_status,
    classify_time_from_open, classify_opening_range_breakout,
    classify_ma_trend, classify_price_vs_ma50, is_opex_week,
)

ET_TZ = ZoneInfo("America/New_York")
TICKER = "NDX"
TEST_DAYS = 10
LOOKBACK = 250

TIME_SLOTS = [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0)]

all_dates = get_available_dates(TICKER, TEST_DAYS + LOOKBACK + 20)
test_dates = all_dates[-TEST_DAYS:]

for test_date in test_dates:
    train_df = build_training_data(TICKER, test_date, LOOKBACK)
    if train_df.empty or len(train_df) < 50:
        continue

    predictor = StatisticalClosePredictor(min_samples=5, **STAT_FEATURE_CONFIG)
    predictor.fit(train_df)

    test_df = load_csv_data(TICKER, test_date)
    if test_df is None or test_df.empty:
        continue

    hist_ctx = get_historical_context(TICKER, test_date)
    day_1 = hist_ctx.get('day_1', {})
    day_2 = hist_ctx.get('day_2', {})
    day_5 = hist_ctx.get('day_5', {})

    prev_close = day_1.get('close')
    if prev_close is None:
        continue

    day_open = get_day_open(test_df)
    actual_close = get_day_close(test_df)
    fh_high, fh_low = get_first_hour_range(test_df)
    or_high, or_low = get_opening_range(test_df)
    price_945 = get_price_at_time(test_df, 9, 45)

    day_ctx = DayContext(
        prev_close=prev_close, day_open=day_open,
        vix1d=get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime()),
        prev_day_close=day_2.get('close'), prev_vix1d=day_1.get('vix1d'),
        prev_day_high=day_1.get('high'), prev_day_low=day_1.get('low'),
        close_5days_ago=day_5.get('close'),
        first_hour_high=fh_high, first_hour_low=fh_low,
        opening_range_high=or_high, opening_range_low=or_low,
        price_at_945=price_945,
        ma5=hist_ctx.get('ma5'), ma10=hist_ctx.get('ma10'),
        ma20=hist_ctx.get('ma20'), ma50=hist_ctx.get('ma50'),
    )

    move = actual_close - prev_close
    pct = move / prev_close * 100
    print(f"\n{'='*120}")
    print(f" {test_date}  |  Prev Close: {prev_close:,.0f}  |  Actual Close: {actual_close:,.0f}  ({move:+,.0f} / {pct:+.2f}%)")
    if day_ctx.ma5:
        print(f"  MAs: 5d={day_ctx.ma5:,.0f}  10d={day_ctx.ma10:,.0f}  20d={day_ctx.ma20:,.0f}  50d={'N/A' if day_ctx.ma50 is None else f'{day_ctx.ma50:,.0f}'}")
        if day_ctx.ma5 and day_ctx.ma10 and day_ctx.ma20:
            print(f"  MA Trend: {classify_ma_trend(day_ctx.ma5, day_ctx.ma10, day_ctx.ma20).value}", end="")
        if day_ctx.ma50:
            print(f"  |  Price vs MA50: {classify_price_vs_ma50(prev_close, day_ctx.ma50).value}", end="")
        print()
    print(f"{'='*120}")

    for hour_et, minute_et in TIME_SLOTS:
        dt = datetime.strptime(test_date, "%Y-%m-%d")
        pred_time = datetime(dt.year, dt.month, dt.day, hour_et, minute_et, tzinfo=ET_TZ)
        target_utc = pred_time.astimezone(ZoneInfo("UTC"))

        before = test_df[test_df['timestamp'] <= target_utc]
        if before.empty:
            continue

        current_price = before.iloc[-1]['close']
        day_high = before['high'].max()
        day_low = before['low'].min()

        ctx = PredictionContext(
            ticker=f"I:{TICKER}", current_price=current_price,
            prev_close=day_ctx.prev_close, day_open=day_ctx.day_open,
            current_time=pred_time, vix1d=day_ctx.vix1d or 15.0,
            day_high=day_high, day_low=day_low,
            prev_day_close=day_ctx.prev_day_close, prev_vix1d=day_ctx.prev_vix1d,
            prev_day_high=day_ctx.prev_day_high, prev_day_low=day_ctx.prev_day_low,
            close_5days_ago=day_ctx.close_5days_ago,
            first_hour_high=day_ctx.first_hour_high, first_hour_low=day_ctx.first_hour_low,
            opening_range_high=day_ctx.opening_range_high,
            opening_range_low=day_ctx.opening_range_low,
            price_at_945=day_ctx.price_at_945,
            ma5=day_ctx.ma5, ma10=day_ctx.ma10, ma20=day_ctx.ma20, ma50=day_ctx.ma50,
        )

        # Build features exactly like predict() does
        features = BucketFeatures(hour=ctx.hour_et, vix_regime=ctx.vix_regime, gap_type=ctx.gap_type)
        feat_details = []
        feat_details.append(f"VIX={ctx.vix_regime.value}")
        feat_details.append(f"Gap={ctx.gap_type.value}")

        if STAT_FEATURE_CONFIG.get('use_intraday_move'):
            im = classify_intraday_move(ctx.intraday_move_pct)
            features.intraday_move = im
            feat_details.append(f"IntraMove={im.value}")
        if STAT_FEATURE_CONFIG.get('use_prior_day_move') and ctx.prior_day_move_pct is not None:
            pdm = classify_prior_day_move(ctx.prior_day_move_pct)
            features.prior_day_move = pdm
            feat_details.append(f"PriorDay={pdm.value}")
        if STAT_FEATURE_CONFIG.get('use_intraday_range') and ctx.intraday_range_pct is not None:
            ir = classify_intraday_range(ctx.intraday_range_pct)
            features.intraday_range = ir
            feat_details.append(f"Range={ir.value}")
        if STAT_FEATURE_CONFIG.get('use_vix_change') and ctx.vix_change_pct is not None:
            vc = classify_vix_change(ctx.vix_change_pct)
            features.vix_change = vc
            feat_details.append(f"VIXChg={vc.value}")
        if STAT_FEATURE_CONFIG.get('use_prior_close_pos') and ctx.prior_close_position is not None:
            pcp = classify_prior_close_position(ctx.prior_close_position)
            features.prior_close_pos = pcp
            feat_details.append(f"PriorPos={pcp.value}")
        if STAT_FEATURE_CONFIG.get('use_momentum_5day') and ctx.momentum_5day_pct is not None:
            m5 = classify_momentum_5day(ctx.momentum_5day_pct)
            features.momentum_5day = m5
            feat_details.append(f"Mom5d={m5.value}")
        if STAT_FEATURE_CONFIG.get('use_first_hour_range') and ctx.first_hour_range_pct is not None:
            fhr = classify_first_hour_range(ctx.first_hour_range_pct)
            features.first_hour_range = fhr
            feat_details.append(f"FHRange={fhr.value}")
        if STAT_FEATURE_CONFIG.get('use_opex'):
            opex = ctx.is_opex_week
            features.is_opex = opex
            feat_details.append(f"OpEx={'Y' if opex else 'N'}")
        if STAT_FEATURE_CONFIG.get('use_opening_drive') and ctx.opening_drive_pct is not None:
            od = classify_opening_drive(ctx.opening_drive_pct)
            features.opening_drive = od
            feat_details.append(f"OD={od.value}")
        if STAT_FEATURE_CONFIG.get('use_gap_fill'):
            gf = ctx.gap_fill_status
            features.gap_fill_status = gf
            feat_details.append(f"GapFill={gf.value}")
        if STAT_FEATURE_CONFIG.get('use_time_period'):
            tp = ctx.time_period
            features.time_from_open = tp
            feat_details.append(f"Time={tp.value}")
        if STAT_FEATURE_CONFIG.get('use_orb') and ctx.orb_status is not None:
            orb = ctx.orb_status
            features.orb_status = orb
            feat_details.append(f"ORB={orb.value}")
        if STAT_FEATURE_CONFIG.get('use_ma_trend') and ctx.ma_trend is not None:
            features.ma_trend = ctx.ma_trend
            feat_details.append(f"MATrend={ctx.ma_trend.value}")
        if STAT_FEATURE_CONFIG.get('use_price_vs_ma50') and ctx.price_vs_ma50 is not None:
            features.price_vs_ma50 = ctx.price_vs_ma50
            feat_details.append(f"PvsMA50={ctx.price_vs_ma50.value}")

        # Morning mode adjustments
        effective_config = STAT_FEATURE_CONFIG.copy()
        if STAT_FEATURE_CONFIG.get('morning_mode') and ctx.is_first_hour:
            effective_config['use_first_hour_range'] = False
            effective_config['use_orb'] = False
            effective_config['use_intraday_range'] = False

        exact_key = features.to_key(effective_config)
        match_type = "EXACT" if exact_key in predictor.percentiles else "FALLBACK"
        samples = predictor.sample_counts.get(exact_key, 0)

        try:
            prediction = predictor.predict(ctx)
            err = actual_close - prediction.predicted_close_mid
            err_pct = err / actual_close * 100
            in_range = prediction.predicted_close_low <= actual_close <= prediction.predicted_close_high

            print(f"\n  {hour_et}:{minute_et:02d} ET  |  Price: {current_price:,.0f}  |  Pred: [{prediction.predicted_close_low:,.0f} - {prediction.predicted_close_high:,.0f}]  Mid: {prediction.predicted_close_mid:,.0f}  |  Err: {err:+,.0f} ({err_pct:+.2f}%)  |  {'HIT' if in_range else 'MISS'}  |  {match_type} ({samples} samples)")
            print(f"    Features: {' | '.join(feat_details)}")
        except Exception as e:
            print(f"\n  {hour_et}:{minute_et:02d} ET  |  Price: {current_price:,.0f}  |  FAILED: {e}")
            print(f"    Features: {' | '.join(feat_details)}")
