"""
Walk-forward backtest comparing percentile, statistical, and combined models.
"""

from datetime import datetime
from typing import List

from .models import ET_TZ, UNIFIED_BAND_NAMES, _intraday_vol_cache
from .bands import combine_bands
from .features import detect_reversal_strength, get_intraday_vol_factor
from .prediction import _train_statistical, compute_percentile_prediction, compute_statistical_prediction
from .display import print_backtest_results

from scripts.percentile_range_backtest import (
    collect_all_data,
    HOURS_TO_CLOSE,
    HOURLY_LABELS,
    TIME_LABELS,
)
from scripts.csv_prediction_backtest import (
    load_csv_data,
    get_available_dates,
    get_day_close,
    get_day_open,
    get_previous_close,
    get_vix1d_at_time,
    get_first_hour_range,
    get_opening_range,
    get_price_at_time,
    get_historical_context,
    get_day_high_low,
    DayContext,
)


def run_backtest(
    ticker: str,
    lookback: int = 250,
    test_days: int = 10,
    vol_scale: bool = True,
    all_slots: bool = False,
):
    """Walk-forward backtest comparing both models + combined."""
    display_ticker = ticker.replace("I:", "") if ticker.startswith("I:") else ticker
    mode = "vol-scaled" if vol_scale else "raw"
    print(f"\nLoading {display_ticker} data ({mode})...")

    needed = lookback + test_days + 10
    all_dates = get_available_dates(ticker, needed)
    if len(all_dates) < lookback + 5:
        print(f"Not enough data. Have {len(all_dates)} dates, need at least {lookback + 5}.")
        return

    print(f"Collecting percentile data for {len(all_dates)} dates...")
    pct_df = collect_all_data(ticker, all_dates)
    if pct_df is None or pct_df.empty:
        print("No percentile data collected.")
        return

    unique_dates = sorted(pct_df['date'].unique())
    max_test = len(unique_dates) - lookback
    if max_test < 1:
        print(f"Not enough dates for walk-forward. Have {len(unique_dates)}, need >{lookback}.")
        return

    actual_test = min(test_days, max_test)
    test_date_list = unique_dates[-actual_test:]

    print(f"Training lookback: {lookback} days (~{lookback // 21} months)")
    print(f"Test period: {test_date_list[0]} to {test_date_list[-1]} ({len(test_date_list)} days)")

    display_labels = TIME_LABELS if all_slots else HOURLY_LABELS

    # Clear intraday vol cache for fresh backtest
    _intraday_vol_cache.clear()

    results = []
    for ti, test_date in enumerate(test_date_list):
        print(f"  Testing {test_date} ({ti+1}/{len(test_date_list)})...")

        test_idx = unique_dates.index(test_date)
        pct_train_dates = set(unique_dates[max(0, test_idx - lookback):test_idx])
        train_dates_sorted = unique_dates[max(0, test_idx - lookback):test_idx]

        # Train statistical predictor for this test date
        stat_predictor = _train_statistical(ticker, test_date, lookback)

        # Load test day data for statistical model
        test_df = load_csv_data(ticker, test_date)
        if test_df is None or test_df.empty:
            continue

        actual_close = get_day_close(test_df)
        day_open = get_day_open(test_df)
        day_full_high, day_full_low = get_day_high_low(test_df)
        fh_high, fh_low = get_first_hour_range(test_df)
        or_high, or_low = get_opening_range(test_df)
        price_945 = get_price_at_time(test_df, 9, 45)

        hist_ctx = get_historical_context(ticker, test_date)
        day_1 = hist_ctx.get('day_1', {})
        day_2 = hist_ctx.get('day_2', {})
        day_5 = hist_ctx.get('day_5', {})

        prev_close_val = day_1.get('close')
        if prev_close_val is None:
            prev_close_val = get_previous_close(ticker, test_date)
        if prev_close_val is None:
            continue

        day_ctx = DayContext(
            prev_close=prev_close_val,
            day_open=day_open,
            vix1d=get_vix1d_at_time(test_date, test_df.iloc[0]['timestamp'].to_pydatetime()),
            prev_day_close=day_2.get('close'),
            prev_vix1d=day_1.get('vix1d'),
            prev_day_high=day_1.get('high'),
            prev_day_low=day_1.get('low'),
            close_5days_ago=day_5.get('close'),
            first_hour_high=fh_high,
            first_hour_low=fh_low,
            opening_range_high=or_high,
            opening_range_low=or_low,
            price_at_945=price_945,
        )

        # Get per-slot data from pct_df for this test day
        test_rows = pct_df[pct_df['date'] == test_date]

        for _, row in test_rows.iterrows():
            time_label = row['time']
            if time_label not in display_labels and not all_slots:
                # Only check standard labels unless showing all
                if time_label not in HOURLY_LABELS:
                    continue

            current_price = row['price']
            above = row['above']
            current_vol = row.get('realized_vol')
            actual_move_pct = row['close_move_pct']

            # Build ET datetime for statistical model
            h, m = time_label.split(":")
            pred_time = datetime(
                int(test_date[:4]), int(test_date[5:7]), int(test_date[8:10]),
                int(h), int(m), tzinfo=ET_TZ,
            )

            # Get day high/low up to this time
            day_high_now = current_price
            day_low_now = current_price
            for utc_h_offset in [5, 4]:
                target_utc_h = int(h) + utc_h_offset
                mask = (
                    (test_df['timestamp'].dt.hour < target_utc_h) |
                    ((test_df['timestamp'].dt.hour == target_utc_h) &
                     (test_df['timestamp'].dt.minute <= int(m)))
                )
                before = test_df[mask]
                if not before.empty:
                    day_high_now = before['high'].max()
                    day_low_now = before['low'].min()
                    break

            # Update VIX for this time
            day_ctx.vix1d = get_vix1d_at_time(test_date, pred_time)

            # Compute reversal blend
            reversal_blend = detect_reversal_strength(
                current_price, prev_close_val, day_open, day_high_now, day_low_now,
            )

            # Compute intraday vol factor
            ivol_factor = get_intraday_vol_factor(
                ticker, test_date, time_label, test_df, train_dates_sorted,
            )

            # --- Percentile model bands ---
            pct_bands = compute_percentile_prediction(
                pct_df, time_label, above, current_price, current_vol,
                pct_train_dates, vol_scale,
                reversal_blend=reversal_blend,
                intraday_vol_factor=ivol_factor,
            )

            # --- Statistical model bands ---
            stat_bands, stat_pred = compute_statistical_prediction(
                stat_predictor, ticker, current_price, pred_time,
                day_ctx, day_high_now, day_low_now,
            )

            if pct_bands is None and stat_bands is None:
                continue

            if pct_bands is None:
                pct_bands = {}
            if stat_bands is None:
                stat_bands = {}

            combined = combine_bands(pct_bands, stat_bands, current_price)

            # Check hits
            result = {
                'date': test_date,
                'time': time_label,
                'hrs_left': HOURS_TO_CLOSE.get(time_label, 0),
                'above': above,
                'price': current_price,
                'actual_close': actual_close,
            }

            for bn in UNIFIED_BAND_NAMES:
                # Percentile
                pb = pct_bands.get(bn)
                if pb:
                    result[f'pct_{bn}_hit'] = pb.lo_price <= actual_close <= pb.hi_price
                    result[f'pct_{bn}_width'] = pb.width_pct
                else:
                    result[f'pct_{bn}_hit'] = False
                    result[f'pct_{bn}_width'] = 0

                # Statistical
                sb = stat_bands.get(bn)
                if sb:
                    result[f'stat_{bn}_hit'] = sb.lo_price <= actual_close <= sb.hi_price
                    result[f'stat_{bn}_width'] = sb.width_pct
                else:
                    result[f'stat_{bn}_hit'] = False
                    result[f'stat_{bn}_width'] = 0

                # Combined
                cb = combined.get(bn)
                if cb:
                    result[f'comb_{bn}_hit'] = cb.lo_price <= actual_close <= cb.hi_price
                    result[f'comb_{bn}_width'] = cb.width_pct
                else:
                    result[f'comb_{bn}_hit'] = False
                    result[f'comb_{bn}_width'] = 0

            results.append(result)

    if not results:
        print("No results generated.")
        return

    print_backtest_results(results, ticker, display_labels, actual_test)
