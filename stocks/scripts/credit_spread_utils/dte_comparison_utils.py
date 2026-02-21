"""
DTE comparison analysis engine for credit spread optimization.

Compares selling credit spreads across different days-to-expiration (DTE)
buckets (0DTE vs 3/5/10 DTE) to find the optimal DTE + profit-target-%
combination. Supports same-day exit with overnight hold tracking.

Also analyses:
- Direction: is it better to sell WITH the day's move or AGAINST it?
  (put spread on up day = with flow; call spread on down day = with flow)
- Time-of-day: which entry hours produce the best results?
"""

import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .data_loader import (
    load_multi_dte_data,
    load_split_source_data,
    find_csv_files_in_dir,
    preload_vix1d_series,
    preload_underlying_close_series,
)
from .interval_analyzer import parse_pst_timestamp, round_to_15_minutes
from .spread_builder import (
    build_credit_spreads,
    calculate_option_price,
    parse_percent_beyond,
    parse_max_spread_width,
)
from .timezone_utils import resolve_timezone, format_timestamp
from .price_utils import get_previous_close_price, get_current_day_close_price

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger


def find_intraday_exit(
    spread: Dict[str, Any],
    day_df: pd.DataFrame,
    entry_interval: datetime,
    profit_target_pct: float,
    use_mid: bool = False,
) -> Tuple[bool, Optional[datetime], Optional[float]]:
    """Check if a spread can be closed profitably within the same trading day.

    Exit when: (entry_credit - cost_to_close) >= entry_credit * profit_target_pct / 100
    """
    entry_credit = spread['net_credit']
    target_profit = entry_credit * (profit_target_pct / 100.0)

    short_ticker = spread.get('short_ticker', '')
    long_ticker = spread.get('long_ticker', '')
    short_strike = spread['short_strike']
    long_strike = spread['long_strike']

    subsequent = day_df[day_df['interval'] > entry_interval]
    if subsequent.empty:
        return False, None, None

    intervals = sorted(subsequent['interval'].unique())

    for check_interval in intervals:
        interval_data = subsequent[subsequent['interval'] == check_interval]

        if short_ticker and long_ticker:
            short_rows = interval_data[interval_data['ticker'] == short_ticker]
            long_rows = interval_data[interval_data['ticker'] == long_ticker]
        else:
            opt_type = spread.get('option_type', '')
            short_rows = interval_data[
                (interval_data['strike'] == short_strike) &
                (interval_data['type'].str.lower() == opt_type.lower())
            ]
            long_rows = interval_data[
                (interval_data['strike'] == long_strike) &
                (interval_data['type'].str.lower() == opt_type.lower())
            ]

        if short_rows.empty or long_rows.empty:
            continue

        close_short_price = calculate_option_price(short_rows.iloc[-1], "buy", use_mid)
        close_long_price = calculate_option_price(long_rows.iloc[-1], "sell", use_mid)

        if close_short_price is None or close_long_price is None:
            continue

        cost_to_close = close_short_price - close_long_price
        current_pnl = entry_credit - cost_to_close

        if current_pnl >= target_profit:
            return True, check_interval, current_pnl

    return False, None, None


def _compute_eod_pnl(
    spread: Dict[str, Any],
    day_df: pd.DataFrame,
    use_mid: bool = False,
) -> Optional[float]:
    """Compute P&L at end of day using the last available quotes."""
    short_ticker = spread.get('short_ticker', '')
    long_ticker = spread.get('long_ticker', '')
    short_strike = spread['short_strike']
    long_strike = spread['long_strike']
    entry_credit = spread['net_credit']

    if day_df.empty:
        return None

    last_interval = day_df['interval'].max()
    last_data = day_df[day_df['interval'] == last_interval]

    if short_ticker and long_ticker:
        short_rows = last_data[last_data['ticker'] == short_ticker]
        long_rows = last_data[last_data['ticker'] == long_ticker]
    else:
        opt_type = spread.get('option_type', '')
        short_rows = last_data[
            (last_data['strike'] == short_strike) &
            (last_data['type'].str.lower() == opt_type.lower())
        ]
        long_rows = last_data[
            (last_data['strike'] == long_strike) &
            (last_data['type'].str.lower() == opt_type.lower())
        ]

    if short_rows.empty or long_rows.empty:
        return None

    close_short = calculate_option_price(short_rows.iloc[-1], "buy", use_mid)
    close_long = calculate_option_price(long_rows.iloc[-1], "sell", use_mid)

    if close_short is None or close_long is None:
        return None

    cost_to_close = close_short - close_long
    return entry_credit - cost_to_close


def track_held_position_inmem(
    spread: Dict[str, Any],
    entry_date,
    full_df: pd.DataFrame,
    hold_max_days: int,
    profit_target_pct: float,
    use_mid: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[str], Optional[float], int]:
    """Track a held position through subsequent days using in-memory DataFrame.

    Uses the already-loaded full DataFrame instead of re-reading CSV files.
    """
    entry_credit = spread['net_credit']
    target_profit = entry_credit * (profit_target_pct / 100.0)

    short_ticker = spread.get('short_ticker', '')
    long_ticker = spread.get('long_ticker', '')
    short_strike = spread['short_strike']
    long_strike = spread['long_strike']
    expiration_str = spread.get('expiration', '')

    try:
        if isinstance(expiration_str, str):
            expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d').date()
        else:
            expiration_date = pd.to_datetime(expiration_str).date()
    except Exception:
        if isinstance(entry_date, datetime):
            expiration_date = entry_date.date() + timedelta(days=hold_max_days)
        else:
            expiration_date = entry_date + timedelta(days=hold_max_days)

    entry_date_obj = entry_date.date() if isinstance(entry_date, datetime) else entry_date

    max_search_date = min(
        expiration_date,
        entry_date_obj + timedelta(days=hold_max_days)
    )

    # Filter full_df to the date range we need
    subsequent_df = full_df[
        (full_df['trading_date'] > entry_date_obj) &
        (full_df['trading_date'] <= max_search_date)
    ]

    if subsequent_df.empty:
        max_loss_pnl = -(spread['width'] - entry_credit)
        final_date = min(expiration_date, entry_date_obj + timedelta(days=hold_max_days))
        return final_date.strftime('%Y-%m-%d'), max_loss_pnl, 0

    subsequent_dates = sorted(subsequent_df['trading_date'].unique())

    for day_count, next_date in enumerate(subsequent_dates, 1):
        next_day_df = subsequent_df[subsequent_df['trading_date'] == next_date]

        intervals = sorted(next_day_df['interval'].unique())

        for check_interval in intervals:
            interval_data = next_day_df[next_day_df['interval'] == check_interval]

            if short_ticker and long_ticker:
                short_rows = interval_data[interval_data['ticker'] == short_ticker]
                long_rows = interval_data[interval_data['ticker'] == long_ticker]
            else:
                opt_type = spread.get('option_type', '')
                short_rows = interval_data[
                    (interval_data['strike'] == short_strike) &
                    (interval_data['type'].str.lower() == opt_type.lower())
                ]
                long_rows = interval_data[
                    (interval_data['strike'] == long_strike) &
                    (interval_data['type'].str.lower() == opt_type.lower())
                ]

            if short_rows.empty or long_rows.empty:
                continue

            close_short = calculate_option_price(short_rows.iloc[-1], "buy", use_mid)
            close_long = calculate_option_price(long_rows.iloc[-1], "sell", use_mid)

            if close_short is None or close_long is None:
                continue

            cost_to_close = close_short - close_long
            current_pnl = entry_credit - cost_to_close

            if current_pnl >= target_profit:
                return str(next_date), current_pnl, day_count

        # If expiration day, compute EOD P&L
        if next_date == expiration_date:
            eod_pnl = _compute_eod_pnl(spread, next_day_df, use_mid)
            if eod_pnl is not None:
                return str(next_date), eod_pnl, day_count

    max_loss_pnl = -(spread['width'] - entry_credit)
    final_date = min(expiration_date, entry_date_obj + timedelta(days=hold_max_days))
    return final_date.strftime('%Y-%m-%d'), max_loss_pnl, len(subsequent_dates)


def _classify_flow(opt_type: str, day_direction: str) -> str:
    """Classify whether a spread goes WITH or AGAINST the day's move.

    Put credit spread = bullish bet (profits if price stays up)
    Call credit spread = bearish bet (profits if price stays down)

    With flow:  put spread on up day, call spread on down day
    Against:    put spread on down day, call spread on up day
    """
    if day_direction == 'flat':
        return 'flat'
    if opt_type == 'put':
        return 'with' if day_direction == 'up' else 'against'
    else:  # call
        return 'with' if day_direction == 'down' else 'against'


def _get_entry_hour(interval, output_tz):
    """Get entry hour from interval timestamp in the output timezone."""
    if interval is None:
        return None
    try:
        ts = interval
        if ts.tzinfo is None:
            from datetime import timezone as _tz
            ts = ts.replace(tzinfo=_tz(timedelta(hours=-8)))
        if output_tz is not None:
            local_ts = ts.astimezone(output_tz)
            return local_ts.hour
        return ts.hour
    except Exception:
        return interval.hour if interval else None


async def analyze_dte_comparison(args, logger):
    """Main entry point for DTE comparison analysis.

    Dispatches to v2 (two-phase) or v1 (original) based on args.
    """
    if getattr(args, 'two_phase', False) or getattr(args, 'percent_beyond_percentile', None):
        return await _analyze_dte_comparison_v2(args, logger)
    return await _analyze_dte_comparison_v1(args, logger)


async def _analyze_dte_comparison_v1(args, logger):
    """Original DTE comparison analysis (unchanged).

    Analyses each trading day across DTE buckets, profit targets,
    direction (with/against the move), and entry hour.
    """
    ticker = args.underlying_ticker
    if not ticker:
        logger.error("--ticker is required for dte-comparison mode")
        return 1

    multi_dte_dir = getattr(args, 'multi_dte_dir', 'options_csv_output_full')
    zero_dte_dir = getattr(args, 'zero_dte_dir', 'options_csv_output')
    dte_buckets_str = getattr(args, 'dte_buckets', '0,3,5,10')
    dte_tolerance = getattr(args, 'dte_tolerance', 1)
    exit_profit_pcts_str = getattr(args, 'exit_profit_pcts', '50,60,70,80,90')
    min_volume = getattr(args, 'min_volume', 5)
    hold_max_days_arg = getattr(args, 'hold_max_days', None)
    use_mid = getattr(args, 'use_mid_price', False)
    summary_only = getattr(args, 'summary', False) or getattr(args, 'summary_only', False)

    dte_buckets = tuple(int(x.strip()) for x in dte_buckets_str.split(','))
    exit_profit_pcts = [float(x.strip()) for x in exit_profit_pcts_str.split(',')]

    try:
        percent_beyond_str = getattr(args, 'percent_beyond', None) or '0.015'
        percent_beyond = parse_percent_beyond(percent_beyond_str)
    except ValueError as e:
        logger.error(f"Invalid percent-beyond: {e}")
        return 1

    try:
        max_spread_width_str = getattr(args, 'max_spread_width', '200')
        max_spread_width = parse_max_spread_width(max_spread_width_str)
    except ValueError as e:
        logger.error(f"Invalid max-spread-width: {e}")
        return 1

    risk_cap = getattr(args, 'risk_cap', None)
    min_spread_width = getattr(args, 'min_spread_width', 5.0)
    max_credit_width_ratio = getattr(args, 'max_credit_width_ratio', 0.60)

    output_tz = None
    try:
        output_tz = resolve_timezone(getattr(args, 'output_timezone', 'America/Los_Angeles'))
    except Exception:
        pass

    # Load multi-DTE data
    t_start = time.time()
    print(
        f"Loading options data: 0DTE from {zero_dte_dir}/{ticker}, "
        f">0DTE from {multi_dte_dir}/{ticker}...",
        flush=True,
    )
    try:
        cache_dir = getattr(args, 'cache_dir', '.options_cache')
        no_cache = getattr(args, 'no_data_cache', False)
        df = load_split_source_data(
            zero_dte_dir=zero_dte_dir,
            multi_dte_dir=multi_dte_dir,
            ticker=ticker,
            start_date=getattr(args, 'start_date', None),
            end_date=getattr(args, 'end_date', None),
            dte_buckets=dte_buckets,
            dte_tolerance=dte_tolerance,
            cache_dir=cache_dir,
            no_cache=no_cache,
            logger=logger,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    t_load = time.time() - t_start
    print(f"Loaded {len(df):,} rows across DTE buckets "
          f"{sorted(df['dte_bucket'].unique())} in {t_load:.1f}s", flush=True)

    # Initialize database
    db_path = getattr(args, 'db_path', None)
    if db_path is None:
        db_path = os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '') or os.getenv('QUEST_DB_STRING', '')
    if isinstance(db_path, str) and db_path.startswith('$'):
        db_path = os.environ.get(db_path[1:], None)

    db = StockQuestDB(
        db_path,
        enable_cache=not getattr(args, 'no_cache', False),
        logger=logger,
    )

    option_type_arg = getattr(args, 'option_type', 'both')
    option_types = ['put', 'call'] if option_type_arg == 'both' else [option_type_arg]

    # Pre-compute trading_date column
    df['trading_date'] = df['timestamp'].apply(
        lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
    )
    trading_dates = sorted(df['trading_date'].unique())
    print(f"Analyzing {len(trading_dates)} trading days...", flush=True)

    # Pre-compute entry_hour for all intervals (avoids repeated tz conversion)
    interval_to_hour = {}
    for iv in df['interval'].unique():
        interval_to_hour[iv] = _get_entry_hour(iv, output_tz)

    # Results: keyed by (dte_bucket, profit_target_pct)
    results = defaultdict(list)
    t_analysis_start = time.time()
    spreads_built = 0
    holds_tracked = 0

    try:
        for day_idx, trading_date in enumerate(trading_dates):
            if day_idx % 25 == 0:
                elapsed = time.time() - t_analysis_start
                rate = day_idx / elapsed if elapsed > 0 and day_idx > 0 else 0
                eta = (len(trading_dates) - day_idx) / rate if rate > 0 else 0
                print(f"  Day {day_idx + 1}/{len(trading_dates)}: {trading_date} "
                      f"[{elapsed:.0f}s elapsed, {rate:.1f} days/s, ~{eta:.0f}s remaining, "
                      f"spreads_built={spreads_built}, holds_tracked={holds_tracked}]",
                      flush=True)

            day_df = df[df['trading_date'] == trading_date]

            first_ts = day_df['timestamp'].min()
            prev_close_result = await get_previous_close_price(db, ticker, first_ts, logger)
            if prev_close_result is None:
                logger.debug(f"No prev close for {trading_date}, skipping")
                continue
            prev_close, prev_close_date = prev_close_result

            # Get current day close for direction analysis
            current_close_result = await get_current_day_close_price(db, ticker, first_ts, logger)
            current_close = None
            day_direction = 'unknown'
            day_move_pct = 0.0
            if current_close_result:
                current_close, _ = current_close_result
                day_move_pct = ((current_close - prev_close) / prev_close) * 100
                if current_close > prev_close * 1.0001:
                    day_direction = 'up'
                elif current_close < prev_close * 0.9999:
                    day_direction = 'down'
                else:
                    day_direction = 'flat'

            # Process each DTE bucket
            for dte_bucket in dte_buckets:
                bucket_df = day_df[day_df['dte_bucket'] == dte_bucket]
                if bucket_df.empty:
                    continue

                hold_max = hold_max_days_arg if hold_max_days_arg is not None else max(dte_bucket, 1)

                for opt_type in option_types:
                    # OPTIMIZATION: Instead of scanning ALL intervals, pick
                    # the first interval of each unique hour. This reduces
                    # build_credit_spreads calls from ~26 to ~7 per day-DTE-type.
                    all_intervals = sorted(bucket_df['interval'].unique())
                    if not all_intervals:
                        continue

                    # Group intervals by hour, take first interval per hour
                    seen_hours = set()
                    representative_intervals = []
                    for iv in all_intervals:
                        h = interval_to_hour.get(iv)
                        if h is not None and h not in seen_hours:
                            seen_hours.add(h)
                            representative_intervals.append(iv)

                    if not representative_intervals:
                        representative_intervals = all_intervals[:1]

                    best_spread = None
                    best_entry_interval = None

                    for entry_interval in representative_intervals:
                        interval_data = bucket_df[bucket_df['interval'] == entry_interval]

                        spreads = build_credit_spreads(
                            interval_data,
                            opt_type,
                            prev_close,
                            percent_beyond,
                            min_spread_width,
                            max_spread_width,
                            use_mid,
                            min_contract_price=getattr(args, 'min_contract_price', 0.0),
                            max_credit_width_ratio=max_credit_width_ratio,
                            min_volume=min_volume,
                        )
                        spreads_built += 1

                        if not spreads:
                            continue

                        if risk_cap is not None:
                            spreads = [s for s in spreads
                                       if s['max_loss_per_contract'] > 0
                                       and s['max_loss_per_contract'] <= risk_cap]

                        if not spreads:
                            continue

                        candidate = max(spreads, key=lambda x: x['net_credit'])

                        if best_spread is None or candidate['net_credit'] > best_spread['net_credit']:
                            best_spread = candidate
                            best_entry_interval = entry_interval

                    if best_spread is None:
                        continue

                    best_spread['option_type'] = opt_type

                    num_contracts = 1
                    if risk_cap is not None and best_spread['max_loss_per_contract'] > 0:
                        num_contracts = int(risk_cap / best_spread['max_loss_per_contract'])
                        if num_contracts < 1:
                            continue

                    entry_hour = interval_to_hour.get(best_entry_interval)

                    # Classify with/against flow
                    flow = _classify_flow(opt_type, day_direction)

                    # Test each profit target
                    for pct in exit_profit_pcts:
                        exited, exit_time, exit_pnl = find_intraday_exit(
                            best_spread, bucket_df, best_entry_interval, pct, use_mid,
                        )

                        hold_days = 0
                        exit_date = str(trading_date)
                        same_day_exit = False

                        if exited:
                            same_day_exit = True
                            final_pnl = exit_pnl
                        else:
                            if dte_bucket == 0:
                                eod_pnl = _compute_eod_pnl(best_spread, bucket_df, use_mid)
                                final_pnl = eod_pnl if eod_pnl is not None else best_spread['net_credit']
                            else:
                                eod_pnl = _compute_eod_pnl(best_spread, bucket_df, use_mid)
                                if eod_pnl is not None and eod_pnl > 0:
                                    final_pnl = eod_pnl
                                else:
                                    # Use in-memory DataFrame instead of re-reading CSVs
                                    holds_tracked += 1
                                    tracked_date, tracked_pnl, tracked_days = track_held_position_inmem(
                                        best_spread, trading_date, df,
                                        hold_max, pct, use_mid, logger,
                                    )
                                    if tracked_pnl is not None:
                                        final_pnl = tracked_pnl
                                        hold_days = tracked_days
                                        if tracked_date:
                                            exit_date = tracked_date
                                    elif eod_pnl is not None:
                                        final_pnl = eod_pnl
                                    else:
                                        final_pnl = best_spread['net_credit']

                        total_pnl = final_pnl * num_contracts * 100

                        trade = {
                            'trading_date': str(trading_date),
                            'exit_date': exit_date,
                            'option_type': opt_type,
                            'dte_bucket': dte_bucket,
                            'profit_target_pct': pct,
                            'entry_credit': best_spread['net_credit'],
                            'entry_credit_total': best_spread['net_credit'] * num_contracts * 100,
                            'pnl_per_share': final_pnl,
                            'total_pnl': total_pnl,
                            'num_contracts': num_contracts,
                            'short_strike': best_spread['short_strike'],
                            'long_strike': best_spread['long_strike'],
                            'width': best_spread['width'],
                            'same_day_exit': same_day_exit,
                            'hold_days': hold_days,
                            'win': final_pnl > 0,
                            'prev_close': prev_close,
                            'current_close': current_close,
                            'day_direction': day_direction,
                            'day_move_pct': day_move_pct,
                            'flow': flow,
                            'entry_hour': entry_hour,
                        }
                        results[(dte_bucket, pct)].append(trade)

    finally:
        await db.close()

    t_total = time.time() - t_start
    print(f"\nAnalysis complete in {t_total:.1f}s "
          f"(spreads_built={spreads_built}, holds_tracked={holds_tracked})", flush=True)

    if not results:
        print("No valid DTE comparison trades found matching the criteria.")
        return 0

    generate_dte_comparison_report(results, dte_buckets, exit_profit_pcts, output_tz, summary_only)
    return 0


# ---------------------------------------------------------------------------
#  REPORTING
# ---------------------------------------------------------------------------

def _summarize_trades(trades: List[Dict]) -> Dict[str, Any]:
    """Compute summary stats for a list of trades."""
    if not trades:
        return {'trades': 0}
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    net = sum(t['total_pnl'] for t in trades)
    gains = sum(t['total_pnl'] for t in trades if t['total_pnl'] > 0)
    losses = sum(abs(t['total_pnl']) for t in trades if t['total_pnl'] < 0)
    pf = (gains / losses) if losses > 0 else float('inf')
    same_day = sum(1 for t in trades if t['same_day_exit'])
    avg_hold = sum(t['hold_days'] for t in trades) / n
    return {
        'trades': n,
        'wins': wins,
        'win_rate': (wins / n * 100),
        'net_pnl': net,
        'avg_pnl': net / n,
        'profit_factor': pf,
        'same_day_pct': (same_day / n * 100),
        'avg_hold_days': avg_hold,
        'total_gains': gains,
        'total_losses': losses,
    }


def generate_dte_comparison_report(
    results: Dict[Tuple[int, float], List[Dict]],
    dte_buckets: Tuple[int, ...],
    exit_profit_pcts: List[float],
    output_tz=None,
    summary_only: bool = False,
):
    """Print comprehensive comparison report with direction + hour analysis."""

    # Flatten all trades for cross-cutting analyses
    all_trades = []
    for trades in results.values():
        all_trades.extend(trades)

    # Use the best profit target per DTE for the main tables
    # (direction and hour analysis use the single best pct per DTE)
    best_pct_per_dte = {}
    for dte_bucket in sorted(dte_buckets):
        best_pct = None
        best_net = float('-inf')
        for pct in exit_profit_pcts:
            trades = results.get((dte_bucket, pct), [])
            if not trades:
                continue
            net = sum(t['total_pnl'] for t in trades)
            if net > best_net:
                best_net = net
                best_pct = pct
        if best_pct is not None:
            best_pct_per_dte[dte_bucket] = best_pct

    # ==================== SECTION 1: DTE Overview ====================
    print("\n" + "=" * 115)
    print("DTE COMPARISON ANALYSIS")
    print("=" * 115)

    print(f"\n{'DTE':<8} {'Trades':<8} {'Win%':<8} {'Avg P&L':<12} "
          f"{'Same-Day%':<11} {'Avg Hold':<10} {'Net P&L':<15} "
          f"{'PF':<8} {'Best PT%':<10}")
    print("-" * 115)

    for dte in sorted(best_pct_per_dte.keys()):
        pct = best_pct_per_dte[dte]
        s = _summarize_trades(results[(dte, pct)])
        pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
        print(f"{dte:<8} {s['trades']:<8} {s['win_rate']:<7.1f}% ${s['avg_pnl']:>+9,.0f}   "
              f"{s['same_day_pct']:<10.1f}% {s['avg_hold_days']:<10.1f} ${s['net_pnl']:>+12,.0f}  "
              f"{pf_str:<8} {pct:.0f}%")

    # ==================== SECTION 2: Full DTE x PT Matrix ====================
    print(f"\n{'=' * 115}")
    print("FULL MATRIX: Net P&L by DTE x Profit Target %")
    print(f"{'=' * 115}")

    sorted_pcts = sorted(exit_profit_pcts)
    header = f"{'DTE':<10}"
    for pct in sorted_pcts:
        header += f" {pct:>5.0f}%PT   "
    print(header)
    print("-" * (10 + len(sorted_pcts) * 12))

    for dte in sorted(dte_buckets):
        row = f"{dte:<10}"
        for pct in sorted_pcts:
            trades = results.get((dte, pct), [])
            if not trades:
                row += f" {'N/A':>10} "
            else:
                net = sum(t['total_pnl'] for t in trades)
                row += f" ${net:>+9,.0f} "
        print(row)

        # Win rate sub-row
        row_wr = f"{'':>10}"
        for pct in sorted_pcts:
            trades = results.get((dte, pct), [])
            if not trades:
                row_wr += f" {'':>10} "
            else:
                wr = sum(1 for t in trades if t['win']) / len(trades) * 100
                row_wr += f" {wr:>8.0f}%W "
        print(row_wr)

    # ==================== SECTION 3: DIRECTION ANALYSIS (0DTE only) ====================
    # Focus on 0DTE since user asked about same-day expiring
    print(f"\n{'=' * 115}")
    print("DIRECTION ANALYSIS: With Flow vs Against Flow (0 DTE)")
    print("  With flow  = put spread on UP day / call spread on DOWN day")
    print("  Against    = put spread on DOWN day / call spread on UP day")
    print(f"{'=' * 115}")

    # Use the best profit target for 0 DTE
    dte_0_pct = best_pct_per_dte.get(0)
    if dte_0_pct is not None:
        dte_0_trades = results.get((0, dte_0_pct), [])

        # Group by flow
        flow_groups = defaultdict(list)
        for t in dte_0_trades:
            flow_groups[t['flow']].append(t)

        print(f"\n{'Direction':<12} {'Trades':<8} {'Win%':<8} {'Avg P&L':<12} "
              f"{'Net P&L':<15} {'PF':<8}")
        print("-" * 80)

        for flow_label in ['with', 'against', 'flat']:
            if flow_label not in flow_groups:
                continue
            s = _summarize_trades(flow_groups[flow_label])
            pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
            print(f"{flow_label:<12} {s['trades']:<8} {s['win_rate']:<7.1f}% "
                  f"${s['avg_pnl']:>+9,.0f}   ${s['net_pnl']:>+12,.0f}  {pf_str:<8}")

        # Break down by option type x direction
        print(f"\n  Detailed: Option Type x Day Direction (0 DTE @ {dte_0_pct:.0f}% PT)")
        print(f"  {'Type':<6} {'Day Dir':<10} {'Flow':<10} {'Trades':<8} {'Win%':<8} "
              f"{'Net P&L':<15} {'PF':<8}")
        print(f"  {'-' * 85}")

        for opt_type in ['put', 'call']:
            for direction in ['up', 'down', 'flat']:
                subset = [t for t in dte_0_trades
                          if t['option_type'] == opt_type and t['day_direction'] == direction]
                if not subset:
                    continue
                s = _summarize_trades(subset)
                flow = _classify_flow(opt_type, direction)
                pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
                print(f"  {opt_type:<6} {direction:<10} {flow:<10} {s['trades']:<8} "
                      f"{s['win_rate']:<7.1f}% ${s['net_pnl']:>+12,.0f}  {pf_str:<8}")

    else:
        print("\n  No 0 DTE trades found.")

    # Direction analysis for ALL DTE buckets
    print(f"\n{'=' * 115}")
    print("DIRECTION ANALYSIS: With Flow vs Against Flow (All DTE Buckets)")
    print(f"{'=' * 115}")

    print(f"\n{'DTE':<6} {'Flow':<10} {'Trades':<8} {'Win%':<8} {'Avg P&L':<12} "
          f"{'Net P&L':<15} {'PF':<8}")
    print("-" * 85)

    for dte in sorted(best_pct_per_dte.keys()):
        pct = best_pct_per_dte[dte]
        dte_trades = results.get((dte, pct), [])
        for flow_label in ['with', 'against']:
            subset = [t for t in dte_trades if t['flow'] == flow_label]
            if not subset:
                continue
            s = _summarize_trades(subset)
            pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
            print(f"{dte:<6} {flow_label:<10} {s['trades']:<8} {s['win_rate']:<7.1f}% "
                  f"${s['avg_pnl']:>+9,.0f}   ${s['net_pnl']:>+12,.0f}  {pf_str:<8}")

    # ==================== SECTION 4: TIME-OF-DAY ANALYSIS ====================
    print(f"\n{'=' * 115}")
    print("TIME-OF-DAY ANALYSIS: Entry Hour Performance (0 DTE)")
    print(f"{'=' * 115}")

    if dte_0_pct is not None:
        dte_0_trades = results.get((0, dte_0_pct), [])

        # Group by entry hour
        hour_groups = defaultdict(list)
        for t in dte_0_trades:
            if t['entry_hour'] is not None:
                hour_groups[t['entry_hour']].append(t)

        if hour_groups:
            print(f"\n{'Hour':<8} {'Trades':<8} {'Win%':<8} {'Avg P&L':<12} "
                  f"{'Net P&L':<15} {'PF':<8} {'Same-Day%':<10}")
            print("-" * 90)

            for hour in sorted(hour_groups.keys()):
                s = _summarize_trades(hour_groups[hour])
                pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
                print(f"{hour:>2}:00   {s['trades']:<8} {s['win_rate']:<7.1f}% "
                      f"${s['avg_pnl']:>+9,.0f}   ${s['net_pnl']:>+12,.0f}  "
                      f"{pf_str:<8} {s['same_day_pct']:<9.1f}%")

            # Hour x Flow breakdown
            print(f"\n  Hour x Flow (0 DTE)")
            print(f"  {'Hour':<8} {'With Flow':<20} {'Against Flow':<20}")
            print(f"  {'':>8} {'W% / Net P&L':<20} {'W% / Net P&L':<20}")
            print(f"  {'-' * 56}")

            for hour in sorted(hour_groups.keys()):
                with_trades = [t for t in hour_groups[hour] if t['flow'] == 'with']
                against_trades = [t for t in hour_groups[hour] if t['flow'] == 'against']

                with_str = "—"
                against_str = "—"

                if with_trades:
                    ws = _summarize_trades(with_trades)
                    with_str = f"{ws['win_rate']:.0f}% / ${ws['net_pnl']:>+,.0f}"
                if against_trades:
                    ags = _summarize_trades(against_trades)
                    against_str = f"{ags['win_rate']:.0f}% / ${ags['net_pnl']:>+,.0f}"

                print(f"  {hour:>2}:00   {with_str:<20} {against_str:<20}")

        else:
            print("\n  No entry hour data available.")

    # Time-of-day for all DTE buckets
    print(f"\n{'=' * 115}")
    print("TIME-OF-DAY ANALYSIS: Entry Hour Performance (All DTE Buckets)")
    print(f"{'=' * 115}")

    print(f"\n{'DTE':<6} {'Hour':<8} {'Trades':<8} {'Win%':<8} {'Avg P&L':<12} "
          f"{'Net P&L':<15} {'PF':<8}")
    print("-" * 80)

    for dte in sorted(best_pct_per_dte.keys()):
        pct = best_pct_per_dte[dte]
        dte_trades = results.get((dte, pct), [])
        hour_groups = defaultdict(list)
        for t in dte_trades:
            if t['entry_hour'] is not None:
                hour_groups[t['entry_hour']].append(t)

        for hour in sorted(hour_groups.keys()):
            s = _summarize_trades(hour_groups[hour])
            pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
            print(f"{dte:<6} {hour:>2}:00   {s['trades']:<8} {s['win_rate']:<7.1f}% "
                  f"${s['avg_pnl']:>+9,.0f}   ${s['net_pnl']:>+12,.0f}  {pf_str:<8}")

    # ==================== SECTION 5: OPTIMAL ====================
    best_combo = None
    best_total_pnl = float('-inf')
    for (dte, pct), trades in results.items():
        net = sum(t['total_pnl'] for t in trades)
        if net > best_total_pnl:
            best_total_pnl = net
            best_combo = (dte, pct)

    if best_combo:
        dte, pct = best_combo
        trades = results[best_combo]
        s = _summarize_trades(trades)

        print(f"\n{'=' * 115}")
        print(f"OPTIMAL: {dte} DTE @ {pct:.0f}% profit target")
        print(f"  Net P&L: ${best_total_pnl:>+,.2f}  |  Trades: {s['trades']}  |  "
              f"Win Rate: {s['win_rate']:.1f}%  |  "
              f"Same-Day Exit: {s['same_day_pct']:.1f}%  |  Avg Hold: {s['avg_hold_days']:.1f} days")

        # Optimal breakdown by flow
        with_trades = [t for t in trades if t['flow'] == 'with']
        against_trades = [t for t in trades if t['flow'] == 'against']
        if with_trades:
            ws = _summarize_trades(with_trades)
            print(f"  WITH flow:    {ws['trades']} trades, {ws['win_rate']:.1f}% WR, ${ws['net_pnl']:>+,.0f}")
        if against_trades:
            ags = _summarize_trades(against_trades)
            print(f"  AGAINST flow: {ags['trades']} trades, {ags['win_rate']:.1f}% WR, ${ags['net_pnl']:>+,.0f}")

        print(f"{'=' * 115}")

    return results


# ===========================================================================
#  V2: TWO-PHASE ANALYSIS  (Phase A = raw trades, Phase B = exit evaluation)
# ===========================================================================


@dataclass
class ExitStrategyConfig:
    """Exit strategy parameters for Phase B evaluation."""
    profit_target_pct: Optional[float] = None   # e.g., 50.0
    exit_dte: Optional[int] = None              # Exit when N days remain
    min_vix1d_entry: Optional[float] = None     # Only enter when VIX >= threshold
    max_vix1d_entry: Optional[float] = None     # Only enter when VIX <= threshold
    stop_loss_multiple: Optional[float] = None  # Exit when loss >= N× credit (e.g., 2.0)
    flow_filter: Optional[str] = None           # 'with', 'against', or None (both)

    def label(self) -> str:
        parts = []
        if self.profit_target_pct is not None:
            parts.append(f"PT{self.profit_target_pct:.0f}")
        if self.stop_loss_multiple is not None:
            parts.append(f"SL{self.stop_loss_multiple:.1f}x")
        if self.exit_dte is not None:
            parts.append(f"ExDTE{self.exit_dte}")
        if self.min_vix1d_entry is not None:
            parts.append(f"VIX>={self.min_vix1d_entry:.0f}")
        if self.max_vix1d_entry is not None:
            parts.append(f"VIX<={self.max_vix1d_entry:.0f}")
        if self.flow_filter is not None:
            parts.append(f"flow={self.flow_filter}")
        return '_'.join(parts) if parts else 'hold_to_exp'


@dataclass
class RawTradeStore:
    """Container for Phase A raw trades, serializable to disk."""
    ticker: str
    percentile: int
    lookback_days: int
    trades: List[Dict[str, Any]] = field(default_factory=list)
    # Phase A params that affect trade selection (used for cache key)
    params_hash: str = ''

    @staticmethod
    def _make_params_hash(
        ticker: str, percentile: int, lookback_days: int,
        min_spread_width: float = 5.0, max_spread_width: str = '200',
        risk_cap: Optional[float] = None, max_credit_width_ratio: float = 0.60,
        start_date: str = '', end_date: str = '',
        strike_selection: str = 'max_credit', iron_condor: bool = False,
    ) -> str:
        """Deterministic hash of all Phase A parameters."""
        key = (f"{ticker}_{percentile}_{lookback_days}_"
               f"w{min_spread_width}-{max_spread_width}_"
               f"rc{risk_cap}_cwr{max_credit_width_ratio}_"
               f"ss{strike_selection}_ic{iron_condor}_"
               f"{start_date}_{end_date}")
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def save(self, cache_dir: str = '.options_cache'):
        os.makedirs(cache_dir, exist_ok=True)
        h = self.params_hash or hashlib.sha256(
            f"{self.ticker}_{self.percentile}_{self.lookback_days}_{len(self.trades)}".encode()
        ).hexdigest()[:12]
        path = os.path.join(cache_dir, f"raw_trades_{self.ticker}_p{self.percentile}_{h}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Cached raw trades: {path} ({size_mb:.1f} MB, {len(self.trades)} trades)")
        return path

    @staticmethod
    def load(path: str) -> 'RawTradeStore':
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def find_cache(ticker: str, percentile: int, params_hash: str = '',
                   cache_dir: str = '.options_cache') -> Optional[str]:
        if not os.path.exists(cache_dir):
            return None
        if params_hash:
            exact = f"raw_trades_{ticker}_p{percentile}_{params_hash}.pkl"
            path = os.path.join(cache_dir, exact)
            if os.path.exists(path):
                return path
            return None
        # Fallback to prefix match (legacy)
        prefix = f"raw_trades_{ticker}_p{percentile}_"
        for fn in os.listdir(cache_dir):
            if fn.startswith(prefix) and fn.endswith('.pkl'):
                return os.path.join(cache_dir, fn)
        return None


async def precompute_adaptive_percent_beyond(
    db,
    ticker: str,
    trading_dates: List[date_type],
    dte_buckets: Tuple[int, ...],
    lookback_days: int,
    percentiles: List[int],
    logger=None,
) -> Dict[Tuple[date_type, int, int], Tuple[float, float]]:
    """Pre-compute percent_beyond for all (date, dte_bucket, percentile) combos.

    1. Query db.get_stock_data(ticker, start-lookback, end, 'daily') ONCE
    2. For each DTE bucket, compute rolling N-day returns (window = max(DTE, 1))
    3. For each trading date, compute trailing lookback percentile
    4. Separate up (call) and down (put) moves for asymmetric percent_beyond

    Returns: {(date, dte_bucket, percentile): (put_pct_beyond, call_pct_beyond)}
    """
    if not trading_dates:
        return {}

    min_date = min(trading_dates)
    max_date = max(trading_dates)

    # Query with extra lookback buffer
    start_str = (min_date - timedelta(days=lookback_days + 30)).isoformat()
    end_str = max_date.isoformat()

    if logger:
        logger.info(f"Precomputing adaptive percent_beyond for {ticker} "
                     f"({len(trading_dates)} dates, {len(dte_buckets)} DTE buckets, "
                     f"percentiles={percentiles}, lookback={lookback_days}d)")

    try:
        df = await db.get_stock_data(
            ticker=ticker,
            start_date=start_str,
            end_date=end_str,
            interval='daily',
        )
    except Exception as e:
        if logger:
            logger.error(f"Failed to load daily data for {ticker}: {e}")
        return {}

    if df is None or df.empty:
        if logger:
            logger.warning(f"No daily data for {ticker}")
        return {}

    # Build close series
    if hasattr(df.index, 'date'):
        close_series = df['close'].astype(float)
        dates_index = pd.Series([idx.date() if hasattr(idx, 'date') else idx
                                  for idx in df.index], index=df.index)
    elif 'date' in df.columns:
        df = df.sort_values('date')
        close_series = df['close'].astype(float)
        close_series.index = range(len(df))
        dates_index = pd.Series([
            d.date() if hasattr(d, 'date') else
            (datetime.strptime(d, '%Y-%m-%d').date() if isinstance(d, str) else d)
            for d in df['date']
        ], index=close_series.index)
    else:
        return {}

    # Build date -> index mapping
    date_to_idx = {}
    for i, d in dates_index.items():
        date_to_idx[d] = i

    result = {}
    trading_dates_set = set(trading_dates)

    # First pass: compute 1-day returns for mean-reversion scaling
    # The key insight: N-day p95 per day DECREASES as N increases
    # (mean-reversion). Raw N-day percentiles overstate the risk.
    # We compute both raw N-day and "per-day scaled" variants so
    # Phase B can exploit tighter strikes for multi-day positions.
    prev_close_1d = close_series.shift(1)
    valid_1d = prev_close_1d.notna() & close_series.notna()
    return_pct_1d = pd.Series(np.nan, index=close_series.index)
    return_pct_1d[valid_1d] = (
        (close_series[valid_1d] - prev_close_1d[valid_1d]) / prev_close_1d[valid_1d]
    )

    for dte_bucket in dte_buckets:
        window = max(dte_bucket, 1)

        # Compute N-day returns: (close[t] - close[t-window]) / close[t-window]
        prev_close = close_series.shift(window)
        valid_mask = prev_close.notna() & close_series.notna()
        return_pct = pd.Series(np.nan, index=close_series.index)
        return_pct[valid_mask] = (
            (close_series[valid_mask] - prev_close[valid_mask]) / prev_close[valid_mask]
        )

        for tdate in trading_dates:
            if tdate not in date_to_idx:
                continue
            tidx = date_to_idx[tdate]

            # Find rows within the lookback window
            lookback_start = tdate - timedelta(days=lookback_days)
            lookback_mask = (dates_index >= lookback_start) & (dates_index < tdate)
            lookback_returns = return_pct[lookback_mask].dropna()

            if len(lookback_returns) < 20:
                continue

            up_returns = lookback_returns[lookback_returns > 0]
            down_returns = lookback_returns[lookback_returns < 0]

            # Also compute 1-day percentiles for scaling comparison
            lookback_1d = return_pct_1d[lookback_mask].dropna()
            up_1d = lookback_1d[lookback_1d > 0]
            down_1d = lookback_1d[lookback_1d < 0]

            for pctl in percentiles:
                # Raw N-day percentiles (actual observed N-day moves)
                if len(up_returns) >= 5:
                    call_pct = float(up_returns.quantile(pctl / 100.0))
                else:
                    call_pct = float(lookback_returns.abs().quantile(pctl / 100.0))

                if len(down_returns) >= 5:
                    put_pct = float(down_returns.abs().quantile(pctl / 100.0))
                else:
                    put_pct = float(lookback_returns.abs().quantile(pctl / 100.0))

                # Mean-reversion adjusted: use 1-day percentile * sqrt(N)
                # This is TIGHTER than raw N-day because actual N-day returns
                # grow slower than sqrt(N) due to mean-reversion.
                # We take the MINIMUM of raw and sqrt-scaled — use the
                # tighter strike when the market is mean-reverting, but
                # respect the raw percentile if it's actually tighter
                # (e.g., trending markets where sqrt overstates).
                if window > 1 and len(up_1d) >= 5 and len(down_1d) >= 5:
                    call_1d = float(up_1d.quantile(pctl / 100.0))
                    put_1d = float(down_1d.abs().quantile(pctl / 100.0))
                    sqrt_factor = np.sqrt(window)

                    call_sqrt = call_1d * sqrt_factor
                    put_sqrt = put_1d * sqrt_factor

                    # Use the empirical ratio between actual and sqrt-scaled
                    # to find the "effective" percent_beyond.
                    # This captures the mean-reversion discount.
                    call_pct = min(call_pct, call_sqrt)
                    put_pct = min(put_pct, put_sqrt)

                result[(tdate, dte_bucket, pctl)] = (put_pct, call_pct)

    if logger:
        logger.info(f"Precomputed {len(result)} adaptive percent_beyond entries "
                     f"(with mean-reversion adjustment for multi-day)")
    return result


def _find_best_intraday_pnl(
    spread: Dict[str, Any],
    day_df: pd.DataFrame,
    entry_interval: datetime,
    use_mid: bool = False,
) -> Tuple[float, float]:
    """Find the best (maximum) P&L achievable at any interval during the day.

    Returns: (best_pnl, best_pnl_pct_of_credit)
    """
    entry_credit = spread['net_credit']
    if entry_credit <= 0:
        return 0.0, 0.0

    short_ticker = spread.get('short_ticker', '')
    long_ticker = spread.get('long_ticker', '')
    short_strike = spread['short_strike']
    long_strike = spread['long_strike']

    subsequent = day_df[day_df['interval'] > entry_interval]
    if subsequent.empty:
        return 0.0, 0.0

    best_pnl = 0.0
    intervals = sorted(subsequent['interval'].unique())

    for check_interval in intervals:
        interval_data = subsequent[subsequent['interval'] == check_interval]

        if short_ticker and long_ticker:
            short_rows = interval_data[interval_data['ticker'] == short_ticker]
            long_rows = interval_data[interval_data['ticker'] == long_ticker]
        else:
            opt_type = spread.get('option_type', '')
            short_rows = interval_data[
                (interval_data['strike'] == short_strike) &
                (interval_data['type'].str.lower() == opt_type.lower())
            ]
            long_rows = interval_data[
                (interval_data['strike'] == long_strike) &
                (interval_data['type'].str.lower() == opt_type.lower())
            ]

        if short_rows.empty or long_rows.empty:
            continue

        close_short = calculate_option_price(short_rows.iloc[-1], "buy", use_mid)
        close_long = calculate_option_price(long_rows.iloc[-1], "sell", use_mid)

        if close_short is None or close_long is None:
            continue

        cost_to_close = close_short - close_long
        current_pnl = entry_credit - cost_to_close
        if current_pnl > best_pnl:
            best_pnl = current_pnl

    best_pnl_pct = (best_pnl / entry_credit * 100) if entry_credit > 0 else 0.0
    return best_pnl, best_pnl_pct


def _find_worst_intraday_pnl(
    spread: Dict[str, Any],
    day_df: pd.DataFrame,
    entry_interval: datetime,
    use_mid: bool = False,
) -> Tuple[float, float]:
    """Find the worst (minimum) P&L at any interval during the day.

    Returns: (worst_pnl, worst_pnl_pct_of_credit)
    """
    entry_credit = spread['net_credit']
    if entry_credit <= 0:
        return 0.0, 0.0

    short_ticker = spread.get('short_ticker', '')
    long_ticker = spread.get('long_ticker', '')
    short_strike = spread['short_strike']
    long_strike = spread['long_strike']

    subsequent = day_df[day_df['interval'] > entry_interval]
    if subsequent.empty:
        return 0.0, 0.0

    worst_pnl = 0.0
    intervals = sorted(subsequent['interval'].unique())

    for check_interval in intervals:
        interval_data = subsequent[subsequent['interval'] == check_interval]

        if short_ticker and long_ticker:
            short_rows = interval_data[interval_data['ticker'] == short_ticker]
            long_rows = interval_data[interval_data['ticker'] == long_ticker]
        else:
            opt_type = spread.get('option_type', '')
            short_rows = interval_data[
                (interval_data['strike'] == short_strike) &
                (interval_data['type'].str.lower() == opt_type.lower())
            ]
            long_rows = interval_data[
                (interval_data['strike'] == long_strike) &
                (interval_data['type'].str.lower() == opt_type.lower())
            ]

        if short_rows.empty or long_rows.empty:
            continue

        close_short = calculate_option_price(short_rows.iloc[-1], "buy", use_mid)
        close_long = calculate_option_price(long_rows.iloc[-1], "sell", use_mid)

        if close_short is None or close_long is None:
            continue

        cost_to_close = close_short - close_long
        current_pnl = entry_credit - cost_to_close
        if current_pnl < worst_pnl:
            worst_pnl = current_pnl

    worst_pnl_pct = (worst_pnl / entry_credit * 100) if entry_credit > 0 else 0.0
    return worst_pnl, worst_pnl_pct


def _track_full_lifecycle(
    spread: Dict[str, Any],
    entry_date: date_type,
    full_df: pd.DataFrame,
    vix1d_series: Dict[date_type, float],
    underlying_series: Dict[date_type, float],
    use_mid: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict], Optional[str], Optional[float], str]:
    """Track a position from entry to expiration, recording daily snapshots.

    Unlike track_held_position_inmem, does NOT exit early on profit target.
    Records mark-to-market every day for Phase B evaluation.

    Returns: (daily_snapshots, resolution_date, resolution_pnl, resolution_type)
    """
    entry_credit = spread['net_credit']
    short_ticker = spread.get('short_ticker', '')
    long_ticker = spread.get('long_ticker', '')
    short_strike = spread['short_strike']
    long_strike = spread['long_strike']
    expiration_str = spread.get('expiration', '')

    try:
        if isinstance(expiration_str, str):
            expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d').date()
        else:
            expiration_date = pd.to_datetime(expiration_str).date()
    except Exception:
        expiration_date = entry_date + timedelta(days=30)

    entry_date_obj = entry_date.date() if isinstance(entry_date, datetime) else entry_date

    # Filter full_df to entry_date through expiration
    lifecycle_df = full_df[
        (full_df['trading_date'] >= entry_date_obj) &
        (full_df['trading_date'] <= expiration_date)
    ]

    if lifecycle_df.empty:
        max_loss = -(spread['width'] - entry_credit)
        return [], expiration_date.isoformat(), max_loss, 'no_data'

    lifecycle_dates = sorted(lifecycle_df['trading_date'].unique())

    daily_snapshots = []
    last_valid_pnl = None
    last_valid_date = None

    for day_offset, trade_date in enumerate(lifecycle_dates):
        day_data = lifecycle_df[lifecycle_df['trading_date'] == trade_date]

        # Compute EOD PnL
        eod_pnl = _compute_eod_pnl(spread, day_data, use_mid)

        # Compute best and worst intraday PnL (for entry day, use entry_interval constraint)
        if day_offset == 0:
            # Entry day: only count intervals after entry
            entry_intervals = sorted(day_data['interval'].unique())
            if entry_intervals:
                best_pnl, best_pnl_pct = _find_best_intraday_pnl(
                    spread, day_data, entry_intervals[0], use_mid
                )
                worst_pnl, worst_pnl_pct = _find_worst_intraday_pnl(
                    spread, day_data, entry_intervals[0], use_mid
                )
            else:
                best_pnl, best_pnl_pct = 0.0, 0.0
                worst_pnl, worst_pnl_pct = 0.0, 0.0
        else:
            # Subsequent days: use all intervals
            dummy_early = day_data['interval'].min() - timedelta(minutes=1)
            best_pnl, best_pnl_pct = _find_best_intraday_pnl(
                spread, day_data, dummy_early, use_mid
            )
            worst_pnl, worst_pnl_pct = _find_worst_intraday_pnl(
                spread, day_data, dummy_early, use_mid
            )

        # Days to expiration
        dte_remaining = (expiration_date - trade_date).days

        snapshot = {
            'date': str(trade_date),
            'day_offset': day_offset,
            'eod_pnl': eod_pnl,
            'eod_pnl_pct': (eod_pnl / entry_credit * 100) if eod_pnl is not None and entry_credit > 0 else None,
            'best_intraday_pnl': best_pnl,
            'best_intraday_pct': best_pnl_pct,
            'worst_intraday_pnl': worst_pnl,
            'worst_intraday_pct': worst_pnl_pct,
            'days_to_expiration': dte_remaining,
            'vix1d': vix1d_series.get(trade_date),
            'underlying_close': underlying_series.get(trade_date),
        }
        daily_snapshots.append(snapshot)

        if eod_pnl is not None:
            last_valid_pnl = eod_pnl
            last_valid_date = str(trade_date)

    # Resolution: expiration EOD P&L or max loss
    if last_valid_pnl is not None:
        resolution_pnl = last_valid_pnl
        resolution_date = last_valid_date
        resolution_type = 'expiration' if last_valid_date == str(expiration_date) else 'last_data'
    else:
        resolution_pnl = -(spread['width'] - entry_credit)
        resolution_date = str(expiration_date)
        resolution_type = 'max_loss'

    return daily_snapshots, resolution_date, resolution_pnl, resolution_type


def _build_close_maps(
    underlying_series: Dict[date_type, float],
    trading_dates: List[date_type],
) -> Tuple[Dict[date_type, float], Dict[date_type, float]]:
    """Build prev_close and current_close maps from underlying close series.

    Eliminates per-day DB queries, enabling sync/multiprocessing execution.

    Returns: (prev_close_map, current_close_map)
        prev_close_map: {trading_date: previous_trading_day_close}
        current_close_map: {trading_date: current_day_close}
    """
    sorted_series_dates = sorted(underlying_series.keys())
    prev_close_map = {}
    current_close_map = {}

    for tdate in trading_dates:
        # Current close: exact match
        if tdate in underlying_series:
            current_close_map[tdate] = underlying_series[tdate]

        # Prev close: most recent date strictly before tdate
        for d in reversed(sorted_series_dates):
            if d < tdate:
                prev_close_map[tdate] = underlying_series[d]
                break

    return prev_close_map, current_close_map


# ---------------------------------------------------------------------------
#  Iron Condor helpers
# ---------------------------------------------------------------------------


def _find_best_intraday_pnl_iron_condor(
    put_spread: Dict[str, Any],
    call_spread: Dict[str, Any],
    day_df: pd.DataFrame,
    entry_interval: datetime,
    use_mid: bool = False,
) -> Tuple[float, float]:
    """Find the best (maximum) combined P&L for an iron condor at any interval."""
    put_credit = put_spread['net_credit']
    call_credit = call_spread['net_credit']
    total_credit = put_credit + call_credit
    if total_credit <= 0:
        return 0.0, 0.0

    subsequent = day_df[day_df['interval'] > entry_interval]
    if subsequent.empty:
        return 0.0, 0.0

    best_pnl = 0.0
    intervals = sorted(subsequent['interval'].unique())

    for check_interval in intervals:
        interval_data = subsequent[subsequent['interval'] == check_interval]
        combined_pnl = 0.0
        valid = True

        for spread in [put_spread, call_spread]:
            short_ticker = spread.get('short_ticker', '')
            long_ticker = spread.get('long_ticker', '')
            if short_ticker and long_ticker:
                short_rows = interval_data[interval_data['ticker'] == short_ticker]
                long_rows = interval_data[interval_data['ticker'] == long_ticker]
            else:
                opt_type = spread.get('option_type', '')
                short_rows = interval_data[
                    (interval_data['strike'] == spread['short_strike']) &
                    (interval_data['type'].str.lower() == opt_type.lower())
                ]
                long_rows = interval_data[
                    (interval_data['strike'] == spread['long_strike']) &
                    (interval_data['type'].str.lower() == opt_type.lower())
                ]
            if short_rows.empty or long_rows.empty:
                valid = False
                break
            close_short = calculate_option_price(short_rows.iloc[-1], "buy", use_mid)
            close_long = calculate_option_price(long_rows.iloc[-1], "sell", use_mid)
            if close_short is None or close_long is None:
                valid = False
                break
            cost_to_close = close_short - close_long
            combined_pnl += spread['net_credit'] - cost_to_close

        if valid and combined_pnl > best_pnl:
            best_pnl = combined_pnl

    best_pnl_pct = (best_pnl / total_credit * 100) if total_credit > 0 else 0.0
    return best_pnl, best_pnl_pct


def _find_worst_intraday_pnl_iron_condor(
    put_spread: Dict[str, Any],
    call_spread: Dict[str, Any],
    day_df: pd.DataFrame,
    entry_interval: datetime,
    use_mid: bool = False,
) -> Tuple[float, float]:
    """Find the worst (minimum) combined P&L for an iron condor at any interval."""
    put_credit = put_spread['net_credit']
    call_credit = call_spread['net_credit']
    total_credit = put_credit + call_credit
    if total_credit <= 0:
        return 0.0, 0.0

    subsequent = day_df[day_df['interval'] > entry_interval]
    if subsequent.empty:
        return 0.0, 0.0

    worst_pnl = 0.0
    intervals = sorted(subsequent['interval'].unique())

    for check_interval in intervals:
        interval_data = subsequent[subsequent['interval'] == check_interval]
        combined_pnl = 0.0
        valid = True

        for spread in [put_spread, call_spread]:
            short_ticker = spread.get('short_ticker', '')
            long_ticker = spread.get('long_ticker', '')
            if short_ticker and long_ticker:
                short_rows = interval_data[interval_data['ticker'] == short_ticker]
                long_rows = interval_data[interval_data['ticker'] == long_ticker]
            else:
                opt_type = spread.get('option_type', '')
                short_rows = interval_data[
                    (interval_data['strike'] == spread['short_strike']) &
                    (interval_data['type'].str.lower() == opt_type.lower())
                ]
                long_rows = interval_data[
                    (interval_data['strike'] == spread['long_strike']) &
                    (interval_data['type'].str.lower() == opt_type.lower())
                ]
            if short_rows.empty or long_rows.empty:
                valid = False
                break
            close_short = calculate_option_price(short_rows.iloc[-1], "buy", use_mid)
            close_long = calculate_option_price(long_rows.iloc[-1], "sell", use_mid)
            if close_short is None or close_long is None:
                valid = False
                break
            cost_to_close = close_short - close_long
            combined_pnl += spread['net_credit'] - cost_to_close

        if valid and combined_pnl < worst_pnl:
            worst_pnl = combined_pnl

    worst_pnl_pct = (worst_pnl / total_credit * 100) if total_credit > 0 else 0.0
    return worst_pnl, worst_pnl_pct


def _compute_eod_pnl_iron_condor(
    put_spread: Dict[str, Any],
    call_spread: Dict[str, Any],
    day_df: pd.DataFrame,
    use_mid: bool = False,
) -> Optional[float]:
    """Compute combined EOD P&L for an iron condor."""
    put_pnl = _compute_eod_pnl(put_spread, day_df, use_mid)
    call_pnl = _compute_eod_pnl(call_spread, day_df, use_mid)
    if put_pnl is not None and call_pnl is not None:
        return put_pnl + call_pnl
    if put_pnl is not None:
        return put_pnl
    if call_pnl is not None:
        return call_pnl
    return None


def _track_full_lifecycle_iron_condor(
    put_spread: Dict[str, Any],
    call_spread: Dict[str, Any],
    entry_date: date_type,
    full_df: pd.DataFrame,
    vix1d_series: Dict[date_type, float],
    underlying_series: Dict[date_type, float],
    use_mid: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Dict], Optional[str], Optional[float], str]:
    """Track an iron condor from entry to expiration, recording combined daily snapshots."""
    total_credit = put_spread['net_credit'] + call_spread['net_credit']

    # Use the earliest expiration
    exp_dates = []
    for spread in [put_spread, call_spread]:
        exp_str = spread.get('expiration', '')
        try:
            if isinstance(exp_str, str):
                exp_dates.append(datetime.strptime(exp_str, '%Y-%m-%d').date())
            else:
                exp_dates.append(pd.to_datetime(exp_str).date())
        except Exception:
            pass
    expiration_date = min(exp_dates) if exp_dates else entry_date + timedelta(days=30)

    entry_date_obj = entry_date.date() if isinstance(entry_date, datetime) else entry_date

    lifecycle_df = full_df[
        (full_df['trading_date'] >= entry_date_obj) &
        (full_df['trading_date'] <= expiration_date)
    ]

    if lifecycle_df.empty:
        put_width = put_spread['width']
        call_width = call_spread['width']
        max_loss = -(max(put_width, call_width) - total_credit)
        return [], expiration_date.isoformat(), max_loss, 'no_data'

    lifecycle_dates = sorted(lifecycle_df['trading_date'].unique())

    daily_snapshots = []
    last_valid_pnl = None
    last_valid_date = None

    for day_offset, trade_date in enumerate(lifecycle_dates):
        day_data = lifecycle_df[lifecycle_df['trading_date'] == trade_date]

        eod_pnl = _compute_eod_pnl_iron_condor(put_spread, call_spread, day_data, use_mid)

        if day_offset == 0:
            entry_intervals = sorted(day_data['interval'].unique())
            if entry_intervals:
                best_pnl, best_pnl_pct = _find_best_intraday_pnl_iron_condor(
                    put_spread, call_spread, day_data, entry_intervals[0], use_mid
                )
                worst_pnl, worst_pnl_pct = _find_worst_intraday_pnl_iron_condor(
                    put_spread, call_spread, day_data, entry_intervals[0], use_mid
                )
            else:
                best_pnl, best_pnl_pct = 0.0, 0.0
                worst_pnl, worst_pnl_pct = 0.0, 0.0
        else:
            dummy_early = day_data['interval'].min() - timedelta(minutes=1)
            best_pnl, best_pnl_pct = _find_best_intraday_pnl_iron_condor(
                put_spread, call_spread, day_data, dummy_early, use_mid
            )
            worst_pnl, worst_pnl_pct = _find_worst_intraday_pnl_iron_condor(
                put_spread, call_spread, day_data, dummy_early, use_mid
            )

        dte_remaining = (expiration_date - trade_date).days

        snapshot = {
            'date': str(trade_date),
            'day_offset': day_offset,
            'eod_pnl': eod_pnl,
            'eod_pnl_pct': (eod_pnl / total_credit * 100) if eod_pnl is not None and total_credit > 0 else None,
            'best_intraday_pnl': best_pnl,
            'best_intraday_pct': best_pnl_pct,
            'worst_intraday_pnl': worst_pnl,
            'worst_intraday_pct': worst_pnl_pct,
            'days_to_expiration': dte_remaining,
            'vix1d': vix1d_series.get(trade_date),
            'underlying_close': underlying_series.get(trade_date),
        }
        daily_snapshots.append(snapshot)

        if eod_pnl is not None:
            last_valid_pnl = eod_pnl
            last_valid_date = str(trade_date)

    if last_valid_pnl is not None:
        resolution_pnl = last_valid_pnl
        resolution_date = last_valid_date
        resolution_type = 'expiration' if last_valid_date == str(expiration_date) else 'last_data'
    else:
        put_width = put_spread['width']
        call_width = call_spread['width']
        resolution_pnl = -(max(put_width, call_width) - total_credit)
        resolution_date = str(expiration_date)
        resolution_type = 'max_loss'

    return daily_snapshots, resolution_date, resolution_pnl, resolution_type


def _build_iron_condor_trade(
    put_spread: Dict[str, Any],
    call_spread: Dict[str, Any],
    trading_date,
    dte_bucket: int,
    prev_close: float,
    current_close: Optional[float],
    day_direction: str,
    day_move_pct: float,
    vix1d_entry: Optional[float],
    percent_beyond: Tuple[float, float],
    risk_cap: Optional[float],
    interval_to_hour: Dict,
    full_df: pd.DataFrame,
    vix1d_series: Dict[date_type, float],
    underlying_series: Dict[date_type, float],
    use_mid: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build an iron condor trade from put and call spreads."""
    total_credit = put_spread['net_credit'] + call_spread['net_credit']
    max_width = max(put_spread['width'], call_spread['width'])

    # For iron condor, max loss = max(put_width, call_width) - total_credit
    # (only one side can be breached at a time)
    max_loss_per_contract = max_width - total_credit
    if max_loss_per_contract <= 0:
        return None

    num_contracts = 1
    if risk_cap is not None and max_loss_per_contract > 0:
        num_contracts = int(risk_cap / (max_loss_per_contract * 100))
        if num_contracts < 1:
            return None

    put_interval = put_spread.get('_entry_interval')
    call_interval = call_spread.get('_entry_interval')
    entry_interval = put_interval or call_interval
    entry_hour = interval_to_hour.get(entry_interval)

    daily_snapshots, resolution_date, resolution_pnl, resolution_type = \
        _track_full_lifecycle_iron_condor(
            put_spread, call_spread, trading_date, full_df,
            vix1d_series, underlying_series, use_mid,
        )

    hold_days = len(daily_snapshots) - 1 if daily_snapshots else 0

    return {
        'trade_id': f"{trading_date}_{dte_bucket}_iron_condor",
        'trading_date': str(trading_date),
        'dte_bucket': dte_bucket,
        'option_type': 'iron_condor',
        'put_leg': {
            'short_strike': put_spread['short_strike'],
            'long_strike': put_spread['long_strike'],
            'width': put_spread['width'],
            'net_credit': put_spread['net_credit'],
            'short_ticker': put_spread.get('short_ticker', ''),
            'long_ticker': put_spread.get('long_ticker', ''),
            'expiration': put_spread.get('expiration', ''),
        },
        'call_leg': {
            'short_strike': call_spread['short_strike'],
            'long_strike': call_spread['long_strike'],
            'width': call_spread['width'],
            'net_credit': call_spread['net_credit'],
            'short_ticker': call_spread.get('short_ticker', ''),
            'long_ticker': call_spread.get('long_ticker', ''),
            'expiration': call_spread.get('expiration', ''),
        },
        'entry_credit': total_credit,
        'short_strike': put_spread['short_strike'],  # Put side for compatibility
        'long_strike': put_spread['long_strike'],
        'width': max_width,
        'short_ticker': put_spread.get('short_ticker', ''),
        'long_ticker': put_spread.get('long_ticker', ''),
        'expiration': put_spread.get('expiration', ''),
        'entry_interval': str(entry_interval) if entry_interval else '',
        'entry_hour': entry_hour,
        'num_contracts': num_contracts,
        'prev_close': prev_close,
        'current_close': current_close,
        'day_direction': day_direction,
        'day_move_pct': day_move_pct,
        'flow': 'neutral',  # Iron condors are direction-neutral
        'vix1d_entry': vix1d_entry,
        'percent_beyond_used': percent_beyond,
        'daily_snapshots': daily_snapshots,
        'resolution_date': resolution_date,
        'resolution_pnl': resolution_pnl,
        'resolution_type': resolution_type,
        'hold_days': hold_days,
    }


# ---------------------------------------------------------------------------
#  Multiprocessing worker for Phase A
# ---------------------------------------------------------------------------
_phase_a_df = None  # Module-level shared DataFrame for worker processes


def _phase_a_worker_init(df_pickle_path: str):
    """Worker initializer: load shared DataFrame from pickle."""
    global _phase_a_df
    _phase_a_df = pd.read_pickle(df_pickle_path)


def _process_days_batch(batch_args: Dict) -> Tuple[List[Dict], int]:
    """Process a batch of trading days for Phase A. Runs in worker process.

    Returns: (list_of_raw_trades, spreads_built_count)
    """
    global _phase_a_df
    df = _phase_a_df

    trading_dates = batch_args['trading_dates']
    dte_buckets = batch_args['dte_buckets']
    percentile = batch_args['percentile']
    adaptive_lookup = batch_args['adaptive_lookup']
    vix1d_series = batch_args['vix1d_series']
    underlying_series = batch_args['underlying_series']
    prev_close_map = batch_args['prev_close_map']
    current_close_map = batch_args['current_close_map']
    interval_to_hour = batch_args['interval_to_hour']
    option_types = batch_args['option_types']
    use_mid = batch_args['use_mid']
    risk_cap = batch_args['risk_cap']
    min_spread_width = batch_args['min_spread_width']
    max_spread_width = batch_args['max_spread_width']
    max_credit_width_ratio = batch_args['max_credit_width_ratio']
    min_volume = batch_args['min_volume']
    min_contract_price = batch_args['min_contract_price']
    strike_selection = batch_args.get('strike_selection', 'max_credit')
    iron_condor = batch_args.get('iron_condor', False)

    trades = []
    spreads_built = 0

    for trading_date in trading_dates:
        day_df = df[df['trading_date'] == trading_date]
        if day_df.empty:
            continue

        prev_close = prev_close_map.get(trading_date)
        if prev_close is None:
            continue

        current_close = current_close_map.get(trading_date)
        day_direction = 'unknown'
        day_move_pct = 0.0
        if current_close is not None:
            day_move_pct = ((current_close - prev_close) / prev_close) * 100
            if current_close > prev_close * 1.0001:
                day_direction = 'up'
            elif current_close < prev_close * 0.9999:
                day_direction = 'down'
            else:
                day_direction = 'flat'

        vix1d_entry = vix1d_series.get(trading_date)

        for dte_bucket in dte_buckets:
            bucket_df = day_df[day_df['dte_bucket'] == dte_bucket]
            if bucket_df.empty:
                continue

            pb_key = (trading_date, dte_bucket, percentile)
            if pb_key in adaptive_lookup:
                put_pct, call_pct = adaptive_lookup[pb_key]
                percent_beyond = (put_pct, call_pct)
            else:
                put_pct, call_pct = 0.015, 0.015
                percent_beyond = (put_pct, call_pct)

            # Initialize iron condor accumulators for this dte_bucket
            _ic_put_spread = None
            _ic_call_spread = None

            for opt_type in option_types:
                all_intervals = sorted(bucket_df['interval'].unique())
                if not all_intervals:
                    continue

                seen_hours = set()
                representative_intervals = []
                for iv in all_intervals:
                    h = interval_to_hour.get(iv)
                    if h is not None and h not in seen_hours:
                        seen_hours.add(h)
                        representative_intervals.append(iv)
                if not representative_intervals:
                    representative_intervals = all_intervals[:1]

                best_spread = None
                best_entry_interval = None

                for entry_interval in representative_intervals:
                    interval_data = bucket_df[bucket_df['interval'] == entry_interval]

                    built = build_credit_spreads(
                        interval_data,
                        opt_type,
                        prev_close,
                        percent_beyond,
                        min_spread_width,
                        max_spread_width,
                        use_mid,
                        min_contract_price=min_contract_price,
                        max_credit_width_ratio=max_credit_width_ratio,
                        min_volume=min_volume,
                    )
                    spreads_built += 1

                    if not built:
                        continue
                    if risk_cap is not None:
                        built = [s for s in built
                                 if s['max_loss_per_contract'] > 0
                                 and s['max_loss_per_contract'] <= risk_cap]
                    if not built:
                        continue

                    if strike_selection == 'boundary':
                        if opt_type == 'put':
                            target_strike = prev_close * (1 - put_pct)
                        else:
                            target_strike = prev_close * (1 + call_pct)
                        candidate = min(built, key=lambda x: abs(x['short_strike'] - target_strike))
                    else:
                        candidate = max(built, key=lambda x: x['net_credit'])

                    if strike_selection == 'boundary':
                        if best_spread is None or abs(candidate['short_strike'] - (prev_close * (1 - put_pct) if opt_type == 'put' else prev_close * (1 + call_pct))) < abs(best_spread['short_strike'] - (prev_close * (1 - put_pct) if opt_type == 'put' else prev_close * (1 + call_pct))):
                            best_spread = candidate
                            best_entry_interval = entry_interval
                    else:
                        if best_spread is None or candidate['net_credit'] > best_spread['net_credit']:
                            best_spread = candidate
                            best_entry_interval = entry_interval

                # Collect per-type best spreads for potential iron condor
                if best_spread is not None:
                    best_spread['option_type'] = opt_type
                    best_spread['_entry_interval'] = best_entry_interval

                if iron_condor:
                    # Store per-type and process iron condor after both types done
                    if opt_type == 'put':
                        _ic_put_spread = best_spread
                        _ic_put_interval = best_entry_interval
                    else:
                        _ic_call_spread = best_spread
                        _ic_call_interval = best_entry_interval
                    continue  # Don't emit individual trades in iron condor mode

                if best_spread is None:
                    continue

                num_contracts = 1
                if risk_cap is not None and best_spread['max_loss_per_contract'] > 0:
                    num_contracts = int(risk_cap / best_spread['max_loss_per_contract'])
                    if num_contracts < 1:
                        continue

                entry_hour = interval_to_hour.get(best_entry_interval)
                flow = _classify_flow(opt_type, day_direction)

                daily_snapshots, resolution_date, resolution_pnl, resolution_type = \
                    _track_full_lifecycle(
                        best_spread, trading_date, df,
                        vix1d_series, underlying_series,
                        use_mid, None,
                    )

                hold_days = len(daily_snapshots) - 1 if daily_snapshots else 0

                raw_trade = {
                    'trade_id': f"{trading_date}_{dte_bucket}_{opt_type}",
                    'trading_date': str(trading_date),
                    'dte_bucket': dte_bucket,
                    'option_type': opt_type,
                    'entry_credit': best_spread['net_credit'],
                    'short_strike': best_spread['short_strike'],
                    'long_strike': best_spread['long_strike'],
                    'width': best_spread['width'],
                    'short_ticker': best_spread.get('short_ticker', ''),
                    'long_ticker': best_spread.get('long_ticker', ''),
                    'expiration': best_spread.get('expiration', ''),
                    'entry_interval': str(best_entry_interval),
                    'entry_hour': entry_hour,
                    'num_contracts': num_contracts,
                    'prev_close': prev_close,
                    'current_close': current_close,
                    'day_direction': day_direction,
                    'day_move_pct': day_move_pct,
                    'flow': flow,
                    'vix1d_entry': vix1d_entry,
                    'percent_beyond_used': percent_beyond,
                    'daily_snapshots': daily_snapshots,
                    'resolution_date': resolution_date,
                    'resolution_pnl': resolution_pnl,
                    'resolution_type': resolution_type,
                    'hold_days': hold_days,
                }
                trades.append(raw_trade)

            # Iron condor: combine put + call after processing both types
            if iron_condor and _ic_put_spread is not None and _ic_call_spread is not None:
                ic_trade = _build_iron_condor_trade(
                    _ic_put_spread, _ic_call_spread, trading_date, dte_bucket,
                    prev_close, current_close, day_direction, day_move_pct,
                    vix1d_entry, percent_beyond, risk_cap,
                    interval_to_hour, df, vix1d_series, underlying_series,
                    use_mid,
                )
                if ic_trade is not None:
                    trades.append(ic_trade)

    return trades, spreads_built


async def build_raw_trades(
    args,
    logger,
    adaptive_lookup: Dict[Tuple[date_type, int, int], Tuple[float, float]],
    percentile: int,
    df: pd.DataFrame,
    db,
    dte_buckets: Tuple[int, ...],
    vix1d_series: Dict[date_type, float],
    underlying_series: Dict[date_type, float],
    output_tz=None,
) -> RawTradeStore:
    """Phase A: Build raw trades with full daily mark-to-market tracking.

    Holds everything to expiration. No profit targets applied here.
    Uses multiprocessing when num_processes > 1 for parallel day processing.
    """
    ticker = args.underlying_ticker
    use_mid = getattr(args, 'use_mid_price', False)
    risk_cap = getattr(args, 'risk_cap', None)
    min_spread_width = getattr(args, 'min_spread_width', 5.0)
    min_volume = getattr(args, 'min_volume', 5)
    max_credit_width_ratio = getattr(args, 'max_credit_width_ratio', 0.60)
    lookback_days = getattr(args, 'percentile_lookback', 180)
    num_processes = getattr(args, 'num_processes', 0)
    strike_selection = getattr(args, 'strike_selection', 'max_credit')
    iron_condor = getattr(args, 'iron_condor', False)

    try:
        max_spread_width_str = getattr(args, 'max_spread_width', '200')
        max_spread_width = parse_max_spread_width(max_spread_width_str)
    except ValueError:
        max_spread_width = (200.0, 200.0)

    option_type_arg = getattr(args, 'option_type', 'both')
    # Iron condor requires both put and call
    if iron_condor:
        option_types = ['put', 'call']
    else:
        option_types = ['put', 'call'] if option_type_arg == 'both' else [option_type_arg]

    # Pre-compute interval-to-hour mapping
    interval_to_hour = {}
    for iv in df['interval'].unique():
        interval_to_hour[iv] = _get_entry_hour(iv, output_tz)

    trading_dates = sorted(df['trading_date'].unique())

    # Compute deterministic cache key from all Phase A parameters
    params_hash = RawTradeStore._make_params_hash(
        ticker=ticker, percentile=percentile, lookback_days=lookback_days,
        min_spread_width=min_spread_width,
        max_spread_width=getattr(args, 'max_spread_width', '200'),
        risk_cap=risk_cap, max_credit_width_ratio=max_credit_width_ratio,
        start_date=getattr(args, 'start_date', ''),
        end_date=getattr(args, 'end_date', ''),
        strike_selection=strike_selection,
        iron_condor=iron_condor,
    )
    store = RawTradeStore(
        ticker=ticker, percentile=percentile, lookback_days=lookback_days,
        params_hash=params_hash,
    )

    # Build close price maps from pre-loaded underlying series (no per-day DB queries)
    prev_close_map, current_close_map = _build_close_maps(underlying_series, trading_dates)

    # If underlying_series is empty (DB was unavailable for preload), fall back to DB queries
    if not prev_close_map:
        logger.warning("No pre-loaded close prices; falling back to per-day DB queries (slower)")
        for tdate in trading_dates:
            day_df = df[df['trading_date'] == tdate]
            if day_df.empty:
                continue
            first_ts = day_df['timestamp'].min()
            result = await get_previous_close_price(db, ticker, first_ts, logger)
            if result:
                prev_close_map[tdate] = result[0]
            result2 = await get_current_day_close_price(db, ticker, first_ts, logger)
            if result2:
                current_close_map[tdate] = result2[0]

    # Determine number of processes
    if num_processes <= 0:
        num_processes = min(mp.cpu_count(), max(1, len(trading_dates) // 4))
    num_processes = max(1, min(num_processes, len(trading_dates)))

    t_start = time.time()
    min_contract_price = getattr(args, 'min_contract_price', 0.0)

    print(f"\n  Phase A: Building raw trades (percentile={percentile}, "
          f"{len(trading_dates)} days, {len(dte_buckets)} DTE buckets, "
          f"{num_processes} processes)...", flush=True)

    if num_processes == 1:
        # Single-process path (no pickle overhead)
        spreads_built = 0
        for day_idx, trading_date in enumerate(trading_dates):
            if day_idx % 50 == 0:
                elapsed = time.time() - t_start
                rate = day_idx / elapsed if elapsed > 0 and day_idx > 0 else 0
                eta = (len(trading_dates) - day_idx) / rate if rate > 0 else 0
                print(f"    Day {day_idx + 1}/{len(trading_dates)}: {trading_date} "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                      f"trades={len(store.trades)}]", flush=True)

            prev_close = prev_close_map.get(trading_date)
            if prev_close is None:
                continue

            current_close = current_close_map.get(trading_date)
            day_direction = 'unknown'
            day_move_pct = 0.0
            if current_close is not None:
                day_move_pct = ((current_close - prev_close) / prev_close) * 100
                if current_close > prev_close * 1.0001:
                    day_direction = 'up'
                elif current_close < prev_close * 0.9999:
                    day_direction = 'down'
                else:
                    day_direction = 'flat'

            day_df = df[df['trading_date'] == trading_date]
            vix1d_entry = vix1d_series.get(trading_date)

            for dte_bucket in dte_buckets:
                bucket_df = day_df[day_df['dte_bucket'] == dte_bucket]
                if bucket_df.empty:
                    continue

                pb_key = (trading_date, dte_bucket, percentile)
                if pb_key in adaptive_lookup:
                    put_pct, call_pct = adaptive_lookup[pb_key]
                    percent_beyond = (put_pct, call_pct)
                else:
                    put_pct, call_pct = 0.015, 0.015
                    percent_beyond = (put_pct, call_pct)

                # Initialize iron condor accumulators for this dte_bucket
                _ic_put_spread_sp = None
                _ic_call_spread_sp = None

                for opt_type in option_types:
                    all_intervals = sorted(bucket_df['interval'].unique())
                    if not all_intervals:
                        continue

                    seen_hours = set()
                    representative_intervals = []
                    for iv in all_intervals:
                        h = interval_to_hour.get(iv)
                        if h is not None and h not in seen_hours:
                            seen_hours.add(h)
                            representative_intervals.append(iv)
                    if not representative_intervals:
                        representative_intervals = all_intervals[:1]

                    best_spread = None
                    best_entry_interval = None

                    for entry_interval in representative_intervals:
                        interval_data = bucket_df[bucket_df['interval'] == entry_interval]

                        spreads = build_credit_spreads(
                            interval_data,
                            opt_type,
                            prev_close,
                            percent_beyond,
                            min_spread_width,
                            max_spread_width,
                            use_mid,
                            min_contract_price=min_contract_price,
                            max_credit_width_ratio=max_credit_width_ratio,
                            min_volume=min_volume,
                        )
                        spreads_built += 1

                        if not spreads:
                            continue
                        if risk_cap is not None:
                            spreads = [s for s in spreads
                                       if s['max_loss_per_contract'] > 0
                                       and s['max_loss_per_contract'] <= risk_cap]
                        if not spreads:
                            continue

                        if strike_selection == 'boundary':
                            if opt_type == 'put':
                                target_strike = prev_close * (1 - put_pct)
                            else:
                                target_strike = prev_close * (1 + call_pct)
                            candidate = min(spreads, key=lambda x: abs(x['short_strike'] - target_strike))
                        else:
                            candidate = max(spreads, key=lambda x: x['net_credit'])

                        if strike_selection == 'boundary':
                            t_strike = prev_close * (1 - put_pct) if opt_type == 'put' else prev_close * (1 + call_pct)
                            if best_spread is None or abs(candidate['short_strike'] - t_strike) < abs(best_spread['short_strike'] - t_strike):
                                best_spread = candidate
                                best_entry_interval = entry_interval
                        else:
                            if best_spread is None or candidate['net_credit'] > best_spread['net_credit']:
                                best_spread = candidate
                                best_entry_interval = entry_interval

                    # Collect per-type best spreads for potential iron condor
                    if best_spread is not None:
                        best_spread['option_type'] = opt_type
                        best_spread['_entry_interval'] = best_entry_interval

                    if iron_condor:
                        # Store per-type and process iron condor after both types done
                        if opt_type == 'put':
                            _ic_put_spread_sp = best_spread
                        else:
                            _ic_call_spread_sp = best_spread
                        continue  # Don't emit individual trades in iron condor mode

                    if best_spread is None:
                        continue

                    num_contracts = 1
                    if risk_cap is not None and best_spread['max_loss_per_contract'] > 0:
                        num_contracts = int(risk_cap / best_spread['max_loss_per_contract'])
                        if num_contracts < 1:
                            continue

                    entry_hour = interval_to_hour.get(best_entry_interval)
                    flow = _classify_flow(opt_type, day_direction)

                    daily_snapshots, resolution_date, resolution_pnl, resolution_type = \
                        _track_full_lifecycle(
                            best_spread, trading_date, df,
                            vix1d_series, underlying_series,
                            use_mid, logger,
                        )

                    hold_days = len(daily_snapshots) - 1 if daily_snapshots else 0

                    raw_trade = {
                        'trade_id': f"{trading_date}_{dte_bucket}_{opt_type}",
                        'trading_date': str(trading_date),
                        'dte_bucket': dte_bucket,
                        'option_type': opt_type,
                        'entry_credit': best_spread['net_credit'],
                        'short_strike': best_spread['short_strike'],
                        'long_strike': best_spread['long_strike'],
                        'width': best_spread['width'],
                        'short_ticker': best_spread.get('short_ticker', ''),
                        'long_ticker': best_spread.get('long_ticker', ''),
                        'expiration': best_spread.get('expiration', ''),
                        'entry_interval': str(best_entry_interval),
                        'entry_hour': entry_hour,
                        'num_contracts': num_contracts,
                        'prev_close': prev_close,
                        'current_close': current_close,
                        'day_direction': day_direction,
                        'day_move_pct': day_move_pct,
                        'flow': flow,
                        'vix1d_entry': vix1d_entry,
                        'percent_beyond_used': percent_beyond,
                        'daily_snapshots': daily_snapshots,
                        'resolution_date': resolution_date,
                        'resolution_pnl': resolution_pnl,
                        'resolution_type': resolution_type,
                        'hold_days': hold_days,
                    }
                    store.trades.append(raw_trade)

                # Iron condor: combine put + call after processing both types
                if iron_condor and _ic_put_spread_sp is not None and _ic_call_spread_sp is not None:
                    ic_trade = _build_iron_condor_trade(
                        _ic_put_spread_sp, _ic_call_spread_sp, trading_date, dte_bucket,
                        prev_close, current_close, day_direction, day_move_pct,
                        vix1d_entry, percent_beyond, risk_cap,
                        interval_to_hour, df, vix1d_series, underlying_series,
                        use_mid,
                    )
                    if ic_trade is not None:
                        store.trades.append(ic_trade)

    else:
        # Multi-process path: save DataFrame to temp file, split days into batches
        tmp_dir = tempfile.mkdtemp(prefix='phase_a_')
        df_path = os.path.join(tmp_dir, 'df.pkl')
        df.to_pickle(df_path)
        print(f"    Saved DataFrame to {df_path} "
              f"({os.path.getsize(df_path) / 1024 / 1024:.1f} MB)", flush=True)

        # Split trading dates into batches
        batch_size = max(1, len(trading_dates) // num_processes)
        batches = []
        for i in range(0, len(trading_dates), batch_size):
            chunk = trading_dates[i:i + batch_size]
            batches.append({
                'trading_dates': chunk,
                'dte_buckets': dte_buckets,
                'percentile': percentile,
                'adaptive_lookup': adaptive_lookup,
                'vix1d_series': vix1d_series,
                'underlying_series': underlying_series,
                'prev_close_map': prev_close_map,
                'current_close_map': current_close_map,
                'interval_to_hour': interval_to_hour,
                'option_types': option_types,
                'use_mid': use_mid,
                'risk_cap': risk_cap,
                'min_spread_width': min_spread_width,
                'max_spread_width': max_spread_width,
                'max_credit_width_ratio': max_credit_width_ratio,
                'min_volume': min_volume,
                'min_contract_price': min_contract_price,
                'strike_selection': strike_selection,
                'iron_condor': iron_condor,
            })

        print(f"    Dispatching {len(batches)} batches to {num_processes} workers...", flush=True)

        try:
            with mp.Pool(
                processes=num_processes,
                initializer=_phase_a_worker_init,
                initargs=(df_path,),
            ) as pool:
                results = pool.map(_process_days_batch, batches)

            spreads_built = 0
            for batch_trades, batch_spreads in results:
                store.trades.extend(batch_trades)
                spreads_built += batch_spreads
        finally:
            # Clean up temp file
            try:
                os.remove(df_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass

    elapsed = time.time() - t_start
    print(f"  Phase A complete: {len(store.trades)} raw trades in {elapsed:.1f}s "
          f"(spreads_built={spreads_built})", flush=True)

    return store


def evaluate_exit_strategies(
    raw_store: RawTradeStore,
    exit_configs: List[ExitStrategyConfig],
    dte_buckets: Tuple[int, ...],
) -> Dict[str, List[Dict[str, Any]]]:
    """Phase B: Apply exit rules to pre-computed raw trades.

    Pure Python, no DB/CSV access. Extremely fast.

    Returns: {exit_config_label: [evaluated_trade_dicts]}
    """
    results = {}

    for config in exit_configs:
        label = config.label()
        evaluated = []

        for raw_trade in raw_store.trades:
            dte_bucket = raw_trade['dte_bucket']
            if dte_bucket not in dte_buckets:
                continue

            # IV filter: skip trade if VIX1D outside bounds
            vix1d_entry = raw_trade.get('vix1d_entry')
            if config.min_vix1d_entry is not None:
                if vix1d_entry is None or vix1d_entry < config.min_vix1d_entry:
                    continue
            if config.max_vix1d_entry is not None:
                if vix1d_entry is None or vix1d_entry > config.max_vix1d_entry:
                    continue

            # Flow filter: skip trades that don't match desired flow direction
            if config.flow_filter is not None:
                if raw_trade.get('flow') != config.flow_filter:
                    continue

            entry_credit = raw_trade['entry_credit']
            snapshots = raw_trade['daily_snapshots']
            num_contracts = raw_trade['num_contracts']
            width = raw_trade['width']

            # Walk through snapshots to find first exit trigger.
            # Priority: stop-loss > profit target > DTE exit > hold to expiration.
            exit_pnl = None
            exit_date = None
            exit_type = None
            hold_days = 0

            for snap in snapshots:
                day_offset = snap['day_offset']

                # Check stop-loss FIRST — stop-loss has highest priority
                if config.stop_loss_multiple is not None:
                    max_loss_threshold = -entry_credit * config.stop_loss_multiple
                    worst_pnl = snap.get('worst_intraday_pnl')
                    if worst_pnl is not None and worst_pnl <= max_loss_threshold:
                        exit_pnl = max_loss_threshold  # Cap at stop-loss level
                        exit_date = snap['date']
                        exit_type = 'stop_loss'
                        hold_days = day_offset
                        break

                # Check profit target — intraday exit is always preferred
                if config.profit_target_pct is not None:
                    if snap['best_intraday_pct'] >= config.profit_target_pct:
                        exit_pnl = entry_credit * (config.profit_target_pct / 100.0)
                        exit_date = snap['date']
                        exit_type = 'profit_target_day0' if day_offset == 0 else 'profit_target'
                        hold_days = day_offset
                        break

                # Check DTE-based exit (only if profit target didn't fire)
                if config.exit_dte is not None:
                    if snap['days_to_expiration'] <= config.exit_dte:
                        # On the DTE exit day, use the better of:
                        # (a) best intraday PnL (if positive), or (b) EOD PnL
                        eod = snap['eod_pnl']
                        best_intra = snap['best_intraday_pnl']
                        if best_intra is not None and eod is not None and best_intra > eod:
                            exit_pnl = best_intra
                        else:
                            exit_pnl = eod
                        exit_date = snap['date']
                        exit_type = 'dte_exit'
                        hold_days = day_offset
                        break

            # If no trigger fired, use resolution (held to expiration)
            if exit_pnl is None:
                exit_pnl = raw_trade['resolution_pnl']
                exit_date = raw_trade['resolution_date']
                exit_type = raw_trade['resolution_type']
                hold_days = raw_trade['hold_days']

            if exit_pnl is None:
                exit_pnl = -(width - entry_credit)

            total_pnl = exit_pnl * num_contracts * 100
            actual_hold = max(hold_days, 1)

            eval_trade = {
                'trade_id': raw_trade['trade_id'],
                'trading_date': raw_trade['trading_date'],
                'exit_date': exit_date,
                'option_type': raw_trade['option_type'],
                'dte_bucket': dte_bucket,
                'entry_credit': entry_credit,
                'entry_credit_total': entry_credit * num_contracts * 100,
                'pnl_per_share': exit_pnl,
                'total_pnl': total_pnl,
                'num_contracts': num_contracts,
                'short_strike': raw_trade['short_strike'],
                'long_strike': raw_trade['long_strike'],
                'width': width,
                'same_day_exit': hold_days == 0,
                'hold_days': hold_days,
                'win': exit_pnl > 0,
                'prev_close': raw_trade['prev_close'],
                'current_close': raw_trade['current_close'],
                'day_direction': raw_trade['day_direction'],
                'day_move_pct': raw_trade['day_move_pct'],
                'flow': raw_trade['flow'],
                'entry_hour': raw_trade['entry_hour'],
                'exit_type': exit_type,
                'vix1d_entry': raw_trade.get('vix1d_entry'),
                'percent_beyond_used': raw_trade.get('percent_beyond_used'),
                # Capital efficiency
                'credit_per_day': entry_credit / actual_hold,
                'daily_roi': (exit_pnl / width / actual_hold * 100) if width > 0 else 0,
                'capital_days_used': width * num_contracts * 100 * actual_hold,
                'annualized_roi': (exit_pnl / width / actual_hold * 252 * 100) if width > 0 and actual_hold > 0 else 0,
                # Raw snapshots for detailed analysis
                'daily_snapshots': snapshots,
            }
            evaluated.append(eval_trade)

        results[label] = evaluated

    return results


async def _analyze_dte_comparison_v2(args, logger):
    """Two-phase DTE comparison analysis.

    Phase A: Build raw trades with full lifecycle tracking (expensive, cached).
    Phase B: Evaluate exit strategies on raw trades (fast, seconds).
    """
    ticker = args.underlying_ticker
    if not ticker:
        logger.error("--ticker is required for dte-comparison mode")
        return 1

    multi_dte_dir = getattr(args, 'multi_dte_dir', 'options_csv_output_full')
    zero_dte_dir = getattr(args, 'zero_dte_dir', 'options_csv_output')
    dte_buckets_str = getattr(args, 'dte_buckets', '0,3,5,10')
    dte_tolerance = getattr(args, 'dte_tolerance', 1)
    use_mid = getattr(args, 'use_mid_price', False)
    lookback_days = getattr(args, 'percentile_lookback', 180)
    cache_dir = getattr(args, 'cache_dir', '.options_cache')

    dte_buckets = tuple(int(x.strip()) for x in dte_buckets_str.split(','))

    # Parse percentile(s)
    pct_beyond_str = getattr(args, 'percent_beyond_percentile', None)
    if pct_beyond_str:
        percentiles = [int(x.strip()) for x in pct_beyond_str.split(',')]
    else:
        percentiles = [95]  # default

    # Parse exit parameters
    exit_profit_pcts_str = getattr(args, 'exit_profit_pcts', '50,70,90')
    exit_profit_pcts = [float(x.strip()) for x in exit_profit_pcts_str.split(',')]

    exit_dte_str = getattr(args, 'exit_dte', None)
    exit_dtes = [None]  # None means hold to expiration
    if exit_dte_str:
        exit_dtes += [int(x.strip()) for x in exit_dte_str.split(',')]

    min_vix1d = getattr(args, 'min_vix1d', None)
    max_vix1d = getattr(args, 'max_vix1d', None)
    stop_loss_multiple = getattr(args, 'stop_loss_multiple', None)
    flow_filter = getattr(args, 'flow_filter', None)

    output_tz = None
    try:
        output_tz = resolve_timezone(getattr(args, 'output_timezone', 'America/Los_Angeles'))
    except Exception:
        pass

    # Load multi-DTE data
    t_start = time.time()
    print(
        f"Loading options data: 0DTE from {zero_dte_dir}/{ticker}, "
        f">0DTE from {multi_dte_dir}/{ticker}...",
        flush=True,
    )
    try:
        no_cache = getattr(args, 'no_data_cache', False)
        df = load_split_source_data(
            zero_dte_dir=zero_dte_dir,
            multi_dte_dir=multi_dte_dir,
            ticker=ticker,
            start_date=getattr(args, 'start_date', None),
            end_date=getattr(args, 'end_date', None),
            dte_buckets=dte_buckets,
            dte_tolerance=dte_tolerance,
            cache_dir=cache_dir,
            no_cache=no_cache,
            logger=logger,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    t_load = time.time() - t_start
    print(f"Loaded {len(df):,} rows across DTE buckets "
          f"{sorted(df['dte_bucket'].unique())} in {t_load:.1f}s", flush=True)

    # Pre-compute trading_date column
    df['trading_date'] = df['timestamp'].apply(
        lambda x: x.date() if hasattr(x, 'date') else pd.to_datetime(x).date()
    )
    trading_dates = sorted(df['trading_date'].unique())
    print(f"Analyzing {len(trading_dates)} trading days...", flush=True)

    # Initialize database
    db_path = getattr(args, 'db_path', None)
    if db_path is None:
        db_path = os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '') or os.getenv('QUEST_DB_STRING', '')
    if isinstance(db_path, str) and db_path.startswith('$'):
        db_path = os.environ.get(db_path[1:], None)

    db = StockQuestDB(
        db_path,
        enable_cache=not getattr(args, 'no_cache', False),
        logger=logger,
    )

    try:
        # Preload VIX1D and underlying close series
        start_str = (min(trading_dates) - timedelta(days=lookback_days + 30)).isoformat()
        end_str = max(trading_dates).isoformat()

        print("Preloading VIX1D and underlying close series...", flush=True)
        vix1d_series = await preload_vix1d_series(db, start_str, end_str, logger)
        underlying_series = await preload_underlying_close_series(db, ticker, start_str, end_str, logger)
        print(f"  VIX1D: {len(vix1d_series)} days, Underlying: {len(underlying_series)} days", flush=True)

        # Precompute adaptive percent_beyond
        print("Precomputing adaptive percent_beyond...", flush=True)
        adaptive_lookup = await precompute_adaptive_percent_beyond(
            db, ticker, trading_dates, dte_buckets, lookback_days, percentiles, logger,
        )
        print(f"  Computed {len(adaptive_lookup)} adaptive entries", flush=True)

        # Phase A: Build raw trades for each percentile
        all_raw_stores = {}
        for pctl in percentiles:
            # Check cache (keyed on all Phase A params)
            strike_sel = getattr(args, 'strike_selection', 'max_credit')
            ic = getattr(args, 'iron_condor', False)
            p_hash = RawTradeStore._make_params_hash(
                ticker=ticker, percentile=pctl, lookback_days=lookback_days,
                min_spread_width=getattr(args, 'min_spread_width', 5.0),
                max_spread_width=getattr(args, 'max_spread_width', '200'),
                risk_cap=getattr(args, 'risk_cap', None),
                max_credit_width_ratio=getattr(args, 'max_credit_width_ratio', 0.60),
                start_date=getattr(args, 'start_date', ''),
                end_date=getattr(args, 'end_date', ''),
                strike_selection=strike_sel,
                iron_condor=ic,
            )
            cached_path = RawTradeStore.find_cache(ticker, pctl, p_hash, cache_dir)
            if cached_path and not getattr(args, 'no_data_cache', False):
                print(f"\n  Loading cached raw trades for p{pctl}: {cached_path}", flush=True)
                raw_store = RawTradeStore.load(cached_path)
                print(f"  Loaded {len(raw_store.trades)} cached trades", flush=True)
            else:
                raw_store = await build_raw_trades(
                    args, logger, adaptive_lookup, pctl,
                    df, db, dte_buckets, vix1d_series, underlying_series, output_tz,
                )
                raw_store.save(cache_dir)

            all_raw_stores[pctl] = raw_store

        # Phase B: Evaluate exit strategies
        print(f"\nPhase B: Evaluating exit strategies...", flush=True)

        # Build exit configs
        exit_configs = []
        for pt in exit_profit_pcts:
            for ed in exit_dtes:
                cfg = ExitStrategyConfig(
                    profit_target_pct=pt,
                    exit_dte=ed,
                    min_vix1d_entry=min_vix1d,
                    max_vix1d_entry=max_vix1d,
                    stop_loss_multiple=stop_loss_multiple,
                    flow_filter=flow_filter,
                )
                exit_configs.append(cfg)

        # Also add a "hold to expiration" config (no profit target, no DTE exit, no VIX filter)
        # This is the unfiltered baseline — VIX filtering only applies to strategy configs
        base_cfg = ExitStrategyConfig()
        exit_configs.insert(0, base_cfg)

        # Add VIX-only configs (no PT, no DTE exit) so Section 4 can compare
        if min_vix1d is not None:
            vix_only_cfg = ExitStrategyConfig(
                min_vix1d_entry=min_vix1d,
                max_vix1d_entry=max_vix1d,
            )
            exit_configs.append(vix_only_cfg)

        all_evaluated = {}  # {(percentile, config_label): [trades]}
        for pctl, raw_store in all_raw_stores.items():
            t_phase_b = time.time()
            evaluated = evaluate_exit_strategies(raw_store, exit_configs, dte_buckets)
            elapsed_b = time.time() - t_phase_b
            for label, trades in evaluated.items():
                all_evaluated[(pctl, label)] = trades
            print(f"  p{pctl}: {len(exit_configs)} configs evaluated in {elapsed_b:.2f}s", flush=True)

    finally:
        await db.close()

    t_total = time.time() - t_start
    print(f"\nTotal analysis time: {t_total:.1f}s", flush=True)

    # Generate v2 report
    generate_dte_comparison_report_v2(
        all_evaluated, all_raw_stores, dte_buckets,
        exit_profit_pcts, exit_dtes, percentiles,
        min_vix1d, max_vix1d, output_tz,
    )

    return 0


# ---------------------------------------------------------------------------
#  V2 REPORTING
# ---------------------------------------------------------------------------

def _summarize_trades_v2(trades: List[Dict]) -> Dict[str, Any]:
    """Compute summary stats for v2 evaluated trades (includes capital efficiency)."""
    if not trades:
        return {'trades': 0}
    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    net = sum(t['total_pnl'] for t in trades)
    gains = sum(t['total_pnl'] for t in trades if t['total_pnl'] > 0)
    losses = sum(abs(t['total_pnl']) for t in trades if t['total_pnl'] < 0)
    pf = (gains / losses) if losses > 0 else float('inf')
    same_day = sum(1 for t in trades if t.get('same_day_exit', False))
    avg_hold = sum(t.get('hold_days', 0) for t in trades) / n

    # Capital efficiency
    avg_credit_per_day = sum(t.get('credit_per_day', 0) for t in trades) / n
    avg_daily_roi = sum(t.get('daily_roi', 0) for t in trades) / n
    avg_annualized = sum(t.get('annualized_roi', 0) for t in trades) / n

    return {
        'trades': n,
        'wins': wins,
        'win_rate': (wins / n * 100),
        'net_pnl': net,
        'avg_pnl': net / n,
        'profit_factor': pf,
        'same_day_pct': (same_day / n * 100),
        'avg_hold_days': avg_hold,
        'total_gains': gains,
        'total_losses': losses,
        'avg_credit_per_day': avg_credit_per_day,
        'avg_daily_roi': avg_daily_roi,
        'avg_annualized_roi': avg_annualized,
    }


def generate_dte_comparison_report_v2(
    all_evaluated: Dict[Tuple[int, str], List[Dict]],
    all_raw_stores: Dict[int, RawTradeStore],
    dte_buckets: Tuple[int, ...],
    exit_profit_pcts: List[float],
    exit_dtes: List[Optional[int]],
    percentiles: List[int],
    min_vix1d: Optional[float],
    max_vix1d: Optional[float],
    output_tz=None,
):
    """Print comprehensive v2 report with 6 sections."""

    print("\n" + "=" * 130)
    print("DTE COMPARISON ANALYSIS v2 — TWO-PHASE (Adaptive Strikes, Theta Decay, IV Filtering, DTE-Based Exits)")
    print("=" * 130)

    # ==================== SECTION 1: DTE Overview + Capital Efficiency ====================
    print(f"\n{'=' * 130}")
    print("SECTION 1: DTE OVERVIEW + CAPITAL EFFICIENCY")
    print(f"{'=' * 130}")

    # For each percentile, find best exit config per DTE
    for pctl in sorted(percentiles):
        print(f"\n  Percentile: p{pctl}")
        print(f"  {'DTE':<6} {'Trades':<8} {'Win%':<8} {'Avg P&L':<12} {'Hold':<8} "
              f"{'Net P&L':<15} {'PF':<8} {'Cr/Day':<10} {'DlyROI':<10} {'AnnROI':<10} {'ExitType':<15}")
        print(f"  {'-' * 125}")

        for dte in sorted(dte_buckets):
            best_label = None
            best_net = float('-inf')

            for label_key, trades in all_evaluated.items():
                p, lbl = label_key
                if p != pctl:
                    continue
                dte_trades = [t for t in trades if t['dte_bucket'] == dte]
                if not dte_trades:
                    continue
                net = sum(t['total_pnl'] for t in dte_trades)
                if net > best_net:
                    best_net = net
                    best_label = label_key

            if best_label is None:
                print(f"  {dte:<6} {'—':>8}")
                continue

            dte_trades = [t for t in all_evaluated[best_label] if t['dte_bucket'] == dte]
            s = _summarize_trades_v2(dte_trades)
            pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
            lbl_str = best_label[1][:14]
            print(f"  {dte:<6} {s['trades']:<8} {s['win_rate']:<7.1f}% ${s['avg_pnl']:>+9,.0f}   "
                  f"{s['avg_hold_days']:<7.1f}d ${s['net_pnl']:>+12,.0f}  {pf_str:<8} "
                  f"${s['avg_credit_per_day']:>.2f}    {s['avg_daily_roi']:>+7.2f}%   "
                  f"{s['avg_annualized_roi']:>+8.1f}%  {lbl_str}")

    # ==================== SECTION 2: DTE × Profit Target Matrix ====================
    print(f"\n{'=' * 130}")
    print("SECTION 2: DTE × PROFIT TARGET MATRIX (Net P&L)")
    print(f"{'=' * 130}")

    for pctl in sorted(percentiles):
        print(f"\n  Percentile: p{pctl}")
        sorted_pcts = sorted(exit_profit_pcts)
        header = f"  {'DTE':<8}"
        header += f"{'HoldExp':>12} "
        for pt in sorted_pcts:
            header += f" {'PT' + str(int(pt)) + '%':>10} "
        print(header)
        print(f"  {'-' * (12 + 12 + len(sorted_pcts) * 12)}")

        for dte in sorted(dte_buckets):
            row = f"  {dte:<8}"

            # Hold to expiration
            hold_key = (pctl, 'hold_to_exp')
            hold_trades = [t for t in all_evaluated.get(hold_key, []) if t['dte_bucket'] == dte]
            if hold_trades:
                net = sum(t['total_pnl'] for t in hold_trades)
                row += f"${net:>+10,.0f} "
            else:
                row += f"{'N/A':>12} "

            for pt in sorted_pcts:
                # Find matching config label
                found = False
                for label_key, trades in all_evaluated.items():
                    p, lbl = label_key
                    if p != pctl:
                        continue
                    if f"PT{pt:.0f}" in lbl and 'ExDTE' not in lbl:
                        dte_trades = [t for t in trades if t['dte_bucket'] == dte]
                        if dte_trades:
                            net = sum(t['total_pnl'] for t in dte_trades)
                            row += f" ${net:>+9,.0f} "
                            found = True
                            break
                if not found:
                    row += f" {'N/A':>10} "
            print(row)

    # ==================== SECTION 3: Theta Decay Curves ====================
    print(f"\n{'=' * 130}")
    print("SECTION 3: THETA DECAY CURVES (Average Spread Value by Day Held)")
    print(f"{'=' * 130}")

    for pctl in sorted(percentiles):
        raw_store = all_raw_stores.get(pctl)
        if not raw_store:
            continue

        print(f"\n  Percentile: p{pctl}")

        for dte in sorted(dte_buckets):
            dte_trades = [t for t in raw_store.trades if t['dte_bucket'] == dte]
            if not dte_trades:
                continue

            # Aggregate snapshots by day_offset
            by_offset = defaultdict(list)
            for trade in dte_trades:
                entry_credit = trade['entry_credit']
                for snap in trade['daily_snapshots']:
                    offset = snap['day_offset']
                    if snap['eod_pnl'] is not None:
                        # Spread value = entry_credit - eod_pnl (cost to close)
                        spread_value = entry_credit - snap['eod_pnl']
                        by_offset[offset].append({
                            'spread_value': spread_value,
                            'eod_pnl': snap['eod_pnl'],
                            'eod_pnl_pct': snap.get('eod_pnl_pct', 0),
                            'entry_credit': entry_credit,
                        })

            if not by_offset:
                continue

            print(f"\n  DTE Bucket: {dte}")
            print(f"  {'Day':<6} {'N':<6} {'AvgValue':<12} {'Avg PnL':<12} {'PnL%':<10} {'Decay%':<10}")
            print(f"  {'-' * 60}")

            # Compute entry-day average for baseline
            day0_data = by_offset.get(0, [])
            baseline_value = None
            if day0_data:
                baseline_value = sum(d['spread_value'] for d in day0_data) / len(day0_data)

            for offset in sorted(by_offset.keys()):
                data = by_offset[offset]
                n = len(data)
                avg_value = sum(d['spread_value'] for d in data) / n
                avg_pnl = sum(d['eod_pnl'] for d in data) / n
                avg_pnl_pct = sum(d.get('eod_pnl_pct', 0) for d in data) / n if data else 0
                decay_pct = 0.0
                if baseline_value and baseline_value > 0:
                    decay_pct = ((baseline_value - avg_value) / baseline_value) * 100

                print(f"  {offset:<6} {n:<6} ${avg_value:<10.2f} ${avg_pnl:<+10.2f} "
                      f"{avg_pnl_pct:<+9.1f}% {decay_pct:<+9.1f}%")

    # ==================== SECTION 4: IV-Conditioned Performance ====================
    print(f"\n{'=' * 130}")
    print("SECTION 4: IV-CONDITIONED PERFORMANCE (VIX1D Ranges)")
    print(f"{'=' * 130}")

    vix_ranges = [(0, 14, 'Low <14'), (14, 18, 'Med 14-18'),
                  (18, 22, 'High 18-22'), (22, 100, 'VHigh 22+')]

    for pctl in sorted(percentiles):
        # Use hold-to-expiration for IV analysis
        hold_key = (pctl, 'hold_to_exp')
        all_trades = all_evaluated.get(hold_key, [])
        if not all_trades:
            continue

        print(f"\n  Percentile: p{pctl}")
        print(f"  {'DTE':<6} {'VIX Range':<14} {'Trades':<8} {'Win%':<8} {'Net P&L':<15} {'PF':<8}")
        print(f"  {'-' * 70}")

        for dte in sorted(dte_buckets):
            dte_trades = [t for t in all_trades if t['dte_bucket'] == dte]
            for lo, hi, label in vix_ranges:
                subset = [t for t in dte_trades
                          if t.get('vix1d_entry') is not None
                          and lo <= t['vix1d_entry'] < hi]
                if not subset:
                    continue
                s = _summarize_trades_v2(subset)
                pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
                print(f"  {dte:<6} {label:<14} {s['trades']:<8} {s['win_rate']:<7.1f}% "
                      f"${s['net_pnl']:>+12,.0f}  {pf_str}")

    # ==================== SECTION 5: DTE-Based Exit Timing ====================
    print(f"\n{'=' * 130}")
    print("SECTION 5: DTE-BASED EXIT TIMING (Net P&L by Exit Point)")
    print(f"{'=' * 130}")

    for pctl in sorted(percentiles):
        print(f"\n  Percentile: p{pctl}")

        # Header
        exit_dte_vals = [ed for ed in exit_dtes if ed is not None]
        header = f"  {'DTE':<8} {'Hold-to-Exp':>14}"
        for ed in sorted(exit_dte_vals):
            header += f" {'Exit@DTE' + str(ed):>14}"
        header += f" {'Best':>10}"
        print(header)
        print(f"  {'-' * (8 + 14 + len(exit_dte_vals) * 15 + 10)}")

        for dte in sorted(dte_buckets):
            row = f"  {dte:<8}"
            best_val = float('-inf')
            best_label = 'N/A'

            # Hold to expiration (any PT that matches)
            hold_key = (pctl, 'hold_to_exp')
            hold_trades = [t for t in all_evaluated.get(hold_key, []) if t['dte_bucket'] == dte]
            if hold_trades:
                net = sum(t['total_pnl'] for t in hold_trades)
                row += f" ${net:>+12,.0f}"
                if net > best_val:
                    best_val = net
                    best_label = 'HoldExp'
            else:
                row += f" {'N/A':>14}"

            for ed in sorted(exit_dte_vals):
                # Find configs with this exit_dte (any PT is fine, use PT50 or first match)
                found = False
                for label_key, trades in all_evaluated.items():
                    p, lbl = label_key
                    if p != pctl:
                        continue
                    if f"ExDTE{ed}" in lbl:
                        dte_trades = [t for t in trades if t['dte_bucket'] == dte]
                        if dte_trades:
                            net = sum(t['total_pnl'] for t in dte_trades)
                            row += f" ${net:>+12,.0f}"
                            if net > best_val:
                                best_val = net
                                best_label = f'DTE{ed}'
                            found = True
                            break
                if not found:
                    row += f" {'N/A':>14}"

            row += f" {best_label:>10}"
            print(row)

    # ==================== SECTION 6: Capital Efficiency Ranking ====================
    print(f"\n{'=' * 130}")
    print("SECTION 6: CAPITAL EFFICIENCY RANKING (All Strategies by Annualized ROI)")
    print(f"{'=' * 130}")

    rankings = []
    for label_key, trades in all_evaluated.items():
        pctl, lbl = label_key
        for dte in sorted(dte_buckets):
            dte_trades = [t for t in trades if t['dte_bucket'] == dte]
            if not dte_trades:
                continue
            s = _summarize_trades_v2(dte_trades)
            rankings.append({
                'percentile': pctl,
                'dte': dte,
                'config': lbl,
                'trades': s['trades'],
                'win_rate': s['win_rate'],
                'net_pnl': s['net_pnl'],
                'avg_hold': s['avg_hold_days'],
                'annualized_roi': s['avg_annualized_roi'],
                'profit_factor': s['profit_factor'],
            })

    rankings.sort(key=lambda x: x['annualized_roi'], reverse=True)

    print(f"\n  {'Rank':<6} {'P%':<6} {'DTE':<6} {'Config':<25} {'Trades':<8} {'Win%':<8} "
          f"{'Net P&L':<15} {'Hold':<8} {'AnnROI':<10} {'PF':<8}")
    print(f"  {'-' * 110}")

    for i, r in enumerate(rankings[:30], 1):
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else 'inf'
        config_str = r['config'][:24]
        print(f"  {i:<6} p{r['percentile']:<4} {r['dte']:<6} {config_str:<25} {r['trades']:<8} "
              f"{r['win_rate']:<7.1f}% ${r['net_pnl']:>+12,.0f} {r['avg_hold']:<7.1f}d "
              f"{r['annualized_roi']:>+8.1f}%  {pf_str}")

    # ==================== SECTION 7: Iron Condor vs Individual Spreads ====================
    # Only show if iron condor trades exist
    ic_trades_exist = any(
        any(t.get('option_type') == 'iron_condor' for t in trades)
        for trades in all_evaluated.values()
    )
    if ic_trades_exist:
        print(f"\n{'=' * 130}")
        print("SECTION 7: IRON CONDOR vs INDIVIDUAL SPREADS")
        print(f"{'=' * 130}")

        for pctl in sorted(percentiles):
            print(f"\n  Percentile: p{pctl}")
            print(f"  {'DTE':<6} {'Type':<14} {'Trades':<8} {'Win%':<8} {'Net P&L':<15} {'PF':<8} {'AnnROI':<10}")
            print(f"  {'-' * 80}")

            for dte in sorted(dte_buckets):
                for label_key, trades in all_evaluated.items():
                    p, lbl = label_key
                    if p != pctl:
                        continue
                    dte_trades = [t for t in trades if t['dte_bucket'] == dte]
                    if not dte_trades:
                        continue

                    for opt_filter, opt_label in [('iron_condor', 'Iron Condor'), ('put', 'Put Spread'), ('call', 'Call Spread')]:
                        subset = [t for t in dte_trades if t.get('option_type') == opt_filter]
                        if not subset:
                            continue
                        s = _summarize_trades_v2(subset)
                        pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
                        print(f"  {dte:<6} {opt_label:<14} {s['trades']:<8} {s['win_rate']:<7.1f}% "
                              f"${s['net_pnl']:>+12,.0f}  {pf_str:<8} {s['avg_annualized_roi']:>+8.1f}%")
                    break  # Only first matching config

    # ==================== SECTION 8: Stop-Loss Impact ====================
    sl_configs = [lbl for _, lbl in all_evaluated.keys() if 'SL' in lbl]
    if sl_configs:
        print(f"\n{'=' * 130}")
        print("SECTION 8: STOP-LOSS IMPACT")
        print(f"{'=' * 130}")

        for pctl in sorted(percentiles):
            print(f"\n  Percentile: p{pctl}")
            print(f"  {'DTE':<6} {'Config':<30} {'Trades':<8} {'Win%':<8} {'Net P&L':<15} {'PF':<8} {'StopOuts':<10}")
            print(f"  {'-' * 100}")

            for dte in sorted(dte_buckets):
                for label_key, trades in sorted(all_evaluated.items(), key=lambda x: x[0][1]):
                    p, lbl = label_key
                    if p != pctl:
                        continue
                    dte_trades = [t for t in trades if t['dte_bucket'] == dte]
                    if not dte_trades:
                        continue
                    s = _summarize_trades_v2(dte_trades)
                    stop_outs = sum(1 for t in dte_trades if t.get('exit_type') == 'stop_loss')
                    pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
                    config_str = lbl[:29]
                    print(f"  {dte:<6} {config_str:<30} {s['trades']:<8} {s['win_rate']:<7.1f}% "
                          f"${s['net_pnl']:>+12,.0f}  {pf_str:<8} {stop_outs}")

    # ==================== SECTION 9: Direction Filter Impact ====================
    flow_configs = [lbl for _, lbl in all_evaluated.keys() if 'flow=' in lbl]
    if flow_configs:
        print(f"\n{'=' * 130}")
        print("SECTION 9: DIRECTION FILTER IMPACT (With-Flow vs Against-Flow vs Unfiltered)")
        print(f"{'=' * 130}")

        for pctl in sorted(percentiles):
            print(f"\n  Percentile: p{pctl}")
            print(f"  {'DTE':<6} {'Filter':<14} {'Trades':<8} {'Win%':<8} {'Net P&L':<15} {'PF':<8} {'AnnROI':<10}")
            print(f"  {'-' * 80}")

            for dte in sorted(dte_buckets):
                # Show unfiltered baseline, then with-flow, then against-flow
                for label_key, trades in all_evaluated.items():
                    p, lbl = label_key
                    if p != pctl:
                        continue
                    dte_trades = [t for t in trades if t['dte_bucket'] == dte]
                    if not dte_trades:
                        continue

                    if 'flow=' in lbl:
                        flow_label = 'with-flow' if 'flow=with' in lbl else 'against-flow'
                    else:
                        flow_label = 'unfiltered'

                    s = _summarize_trades_v2(dte_trades)
                    pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
                    print(f"  {dte:<6} {flow_label:<14} {s['trades']:<8} {s['win_rate']:<7.1f}% "
                          f"${s['net_pnl']:>+12,.0f}  {pf_str:<8} {s['avg_annualized_roi']:>+8.1f}%")

    # ==================== SECTION 10: Percentile Boundary Analysis ====================
    if len(percentiles) > 1:
        print(f"\n{'=' * 130}")
        print("SECTION 10: PERCENTILE BOUNDARY ANALYSIS (Performance by Percentile)")
        print(f"{'=' * 130}")

        # Find the best config label (by net P&L across all) to use for comparison
        print(f"\n  {'P%':<6} {'DTE':<6} {'Trades':<8} {'Win%':<8} {'Net P&L':<15} {'PF':<8} {'AnnROI':<10} {'AvgHold':<10}")
        print(f"  {'-' * 85}")

        for pctl in sorted(percentiles):
            for dte in sorted(dte_buckets):
                # Use hold_to_exp for comparison
                hold_key = (pctl, 'hold_to_exp')
                dte_trades = [t for t in all_evaluated.get(hold_key, []) if t['dte_bucket'] == dte]
                if not dte_trades:
                    continue
                s = _summarize_trades_v2(dte_trades)
                pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else 'inf'
                print(f"  p{pctl:<4} {dte:<6} {s['trades']:<8} {s['win_rate']:<7.1f}% "
                      f"${s['net_pnl']:>+12,.0f}  {pf_str:<8} {s['avg_annualized_roi']:>+8.1f}%   "
                      f"{s['avg_hold_days']:.1f}d")

    print(f"\n{'=' * 130}")


async def run_dte_comparison(args):
    """Entry point called from main(). Validates args and runs analysis."""
    logger = get_logger("dte_comparison", level=getattr(args, 'log_level', 'INFO'))

    if not args.underlying_ticker:
        print("Error: --ticker is required for --mode dte-comparison")
        return 1

    multi_dte_dir = getattr(args, 'multi_dte_dir', 'options_csv_output_full')
    zero_dte_dir = getattr(args, 'zero_dte_dir', 'options_csv_output')
    if not os.path.isdir(multi_dte_dir):
        print(f"Error: Multi-DTE directory not found: {multi_dte_dir}")
        return 1
    if not os.path.isdir(zero_dte_dir):
        print(f"Error: Zero-DTE directory not found: {zero_dte_dir}")
        return 1

    ticker_dir = os.path.join(multi_dte_dir, args.underlying_ticker.upper())
    if not os.path.isdir(ticker_dir):
        print(f"Error: Ticker directory not found: {ticker_dir}")
        return 1
    zero_dte_ticker_dir = os.path.join(zero_dte_dir, args.underlying_ticker.upper())
    if not os.path.isdir(zero_dte_ticker_dir):
        print(f"Error: Zero-DTE ticker directory not found: {zero_dte_ticker_dir}")
        return 1

    return await analyze_dte_comparison(args, logger)
