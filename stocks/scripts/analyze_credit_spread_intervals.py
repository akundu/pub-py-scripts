#!/usr/bin/env python3
"""
Analyze credit spreads from CSV options data at 15-minute intervals.

This program:
1. Reads a CSV file with options data (timestamps in PST)
2. Filters for 0DTE options only (timestamp date == expiration date)
3. Groups data by 15-minute intervals
4. Finds the maximum credit spread for call/put options
5. Filters based on % beyond previous trading day's closing price
6. Caps risk at a specified amount
7. Uses QuestDB to get previous trading day's closing and opening prices

Features:
- Min Trading Hour: Starts counting transactions only after specified hour (optional, uses output timezone)
- Max Trading Hour: Prevents adding positions after specified hour (default: 3PM in output timezone)
- Force Close Hour: Close all positions at specified hour, P&L calculated based on actual spread value
- Multiprocessing: Process multiple files in parallel using multiple CPU cores
- Profit Target: Exit positions early when target profit percentage is reached

Price Data:
- Requires bid/ask prices for option pricing (no day_close fallback)
- For selling: uses bid price (what you receive)
- For buying: uses ask price (what you pay)
- Options without valid bid/ask are skipped
"""

import asyncio
import logging
import sys
import multiprocessing
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import pandas as pd

# Project Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add scripts directory to path for credit_spread_utils imports
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger
from common.market_hours import is_market_hours, compute_market_transition_times

# Import utility modules
from credit_spread_utils.arg_parser import parse_args
from credit_spread_utils.timezone_utils import (
    resolve_timezone,
    format_timestamp,
    normalize_timestamp,
)
from credit_spread_utils.capital_utils import (
    calculate_position_capital,
    filter_results_by_capital_limit,
)
from credit_spread_utils.rate_limiter import SlidingWindowRateLimiter
from credit_spread_utils.time_block_rate_limiter import TimeBlockRateLimiter
from credit_spread_utils.delta_utils import (
    DeltaFilterConfig,
    parse_delta_range,
    format_delta_filter_info,
)

# Import from extracted modules
from credit_spread_utils.spread_builder import (
    parse_percent_beyond,
    parse_max_spread_width,
    parse_min_premium_diff,
)
from credit_spread_utils.interval_analyzer import analyze_interval
from credit_spread_utils.metrics import (
    filter_top_n_per_day,
    print_trading_statistics,
    generate_hourly_histogram,
)
from credit_spread_utils.data_loader import (
    find_csv_files_in_dir,
    load_data_cached,
    clear_cache,
    process_single_csv,
    process_single_csv_sync,
)
from credit_spread_utils.grid_search import run_grid_search
from credit_spread_utils.continuous_runner import (
    run_continuous_analysis,
    _run_single_analysis_iteration,
)
from credit_spread_utils.close_predictor_gate import (
    ClosePredictorGate,
    ClosePredictorGateConfig,
    parse_close_predictor_buffer,
)
from credit_spread_utils.output_formatter import (
    format_best_current_option,
    format_summary_line,
    build_summary_parts,
)

# Strategy framework imports
from credit_spread_utils.strategies import StrategyRegistry, StrategyConfig

# Analysis mode imports
from credit_spread_utils.price_movement_utils import run_price_movement_analysis
from credit_spread_utils.max_move_utils import run_max_move_analysis
from credit_spread_utils.risk_gradient_utils import run_risk_gradient_analysis
from credit_spread_utils.dte_comparison_utils import run_dte_comparison

# Scale-in imports
from credit_spread_utils.scale_in_utils import (
    ScaleInConfig,
    initialize_scale_in_trade,
    calculate_layered_pnl,
    process_price_update,
    generate_scale_in_summary,
    format_scale_in_result,
    load_scale_in_config,
    check_breach,
)

# Tiered investment imports
from credit_spread_utils.tiered_investment_utils import (
    TieredInvestmentConfig,
    initialize_tiered_trade,
    calculate_all_tiers_pnl,
    generate_tiered_summary,
    aggregate_tiered_results,
    print_tiered_statistics,
    load_tiered_config,
)


# ============================================================================
# SCALE-IN ANALYSIS FUNCTIONS
# ============================================================================

async def analyze_scale_in_trade(
    db: StockQuestDB,
    trading_date: datetime,
    option_type: str,
    prev_close: float,
    current_close: float,
    scale_in_config: ScaleInConfig,
    logger: logging.Logger,
    intraday_prices: Optional[List[Tuple[datetime, float]]] = None,
) -> Dict[str, Any]:
    """
    Analyze a scale-in trade for a single day.

    This function simulates the scale-in strategy for a single trading day,
    tracking which layers get triggered based on price movements and
    calculating the final P&L for all layers.

    Args:
        db: Database connection
        trading_date: The trading date
        option_type: 'put' or 'call'
        prev_close: Previous day's closing price
        current_close: Current day's closing price (EOD)
        scale_in_config: Scale-in configuration
        logger: Logger instance
        intraday_prices: Optional list of (timestamp, price) tuples for intraday simulation

    Returns:
        Dictionary with trade state and summary
    """
    # Initialize the trade with all layer positions
    trade_state = initialize_scale_in_trade(
        trading_date=trading_date,
        option_type=option_type,
        prev_close=prev_close,
        config=scale_in_config,
        initial_credit_estimate=3.50,  # Conservative estimate
        logger=logger
    )

    # If we have intraday prices, simulate price updates to trigger layers
    if intraday_prices:
        for price_time, price in intraday_prices:
            trade_state, new_layer_triggered = process_price_update(
                trade_state=trade_state,
                current_price=price,
                current_time=price_time,
                config=scale_in_config,
                logger=logger
            )

            if new_layer_triggered:
                logger.debug(f"Layer triggered at {price_time}: price={price:.2f}")
    else:
        # Without intraday prices, check if layers would have been triggered by EOD price
        layers = scale_in_config.get_layers(option_type)

        for layer_config in layers:
            layer_position = trade_state.get_layer(layer_config.level)
            if layer_position is None:
                continue

            # Skip L1 as it's always triggered at entry
            if layer_config.trigger == 'entry':
                continue

            # Check if previous layer was breached by EOD price
            prev_layer_num = layer_config.level - 1
            prev_layer = trade_state.get_layer(prev_layer_num)

            if prev_layer and prev_layer.triggered:
                if check_breach(option_type, prev_layer.short_strike, current_close):
                    layer_position.triggered = True
                    layer_position.entry_time = trading_date
                    trade_state.current_layer = layer_config.level
                    logger.debug(
                        f"Layer {layer_config.level} triggered (EOD breach): "
                        f"price={current_close:.2f}, L{prev_layer_num} strike={prev_layer.short_strike:.2f}"
                    )

    # Calculate P&L for all triggered layers
    trade_state = calculate_layered_pnl(
        trade_state=trade_state,
        close_price=current_close,
        close_time=trading_date
    )

    # Generate summary
    single_entry_pnl = None
    l1 = trade_state.get_layer(1)
    if l1 and l1.triggered and l1.actual_pnl_total is not None:
        l1_capital_ratio = 0.40  # L1 typically gets 40%
        single_entry_pnl = l1.actual_pnl_total / l1_capital_ratio

    summary = generate_scale_in_summary(trade_state, single_entry_pnl)

    return {
        'trade_state': trade_state,
        'summary': summary,
        'formatted': format_scale_in_result(summary)
    }


def aggregate_scale_in_results(
    scale_in_results: List[Dict[str, Any]],
    output_tz=None
) -> Dict[str, Any]:
    """
    Aggregate scale-in results across multiple trading days.

    Args:
        scale_in_results: List of scale-in trade results
        output_tz: Output timezone for date formatting

    Returns:
        Aggregated statistics dictionary
    """
    if not scale_in_results:
        return {
            'total_trades': 0,
            'total_capital_deployed': 0.0,
            'total_initial_credit': 0.0,
            'total_actual_pnl': 0.0,
            'win_rate': 0.0,
            'avg_layers_triggered': 0.0,
            'avg_layers_breached': 0.0,
            'total_recovery_amount': 0.0,
            'avg_recovery_pct': 0.0,
        }

    total_trades = len(scale_in_results)
    total_capital_deployed = 0.0
    total_initial_credit = 0.0
    total_actual_pnl = 0.0
    total_layers_triggered = 0
    total_layers_breached = 0
    winning_trades = 0
    total_recovery_amount = 0.0
    recovery_count = 0

    layer_stats = {1: {'triggered': 0, 'breached': 0, 'pnl': 0.0},
                   2: {'triggered': 0, 'breached': 0, 'pnl': 0.0},
                   3: {'triggered': 0, 'breached': 0, 'pnl': 0.0}}

    for result in scale_in_results:
        summary = result['summary']

        total_capital_deployed += summary.get('total_capital_deployed', 0.0)
        total_initial_credit += summary.get('total_initial_credit', 0.0)

        pnl = summary.get('total_actual_pnl')
        if pnl is not None:
            total_actual_pnl += pnl
            if pnl > 0:
                winning_trades += 1

        total_layers_triggered += summary.get('num_layers_triggered', 0)
        total_layers_breached += summary.get('num_layers_breached', 0)

        if 'recovery_vs_single' in summary:
            recovery = summary['recovery_vs_single']
            total_recovery_amount += recovery.get('recovery_amount', 0.0)
            recovery_count += 1

        for layer in summary.get('layers', []):
            level = layer.get('level', 0)
            if level in layer_stats:
                if layer.get('triggered'):
                    layer_stats[level]['triggered'] += 1
                if layer.get('breach_detected'):
                    layer_stats[level]['breached'] += 1
                if layer.get('actual_pnl') is not None:
                    layer_stats[level]['pnl'] += layer['actual_pnl']

    testable_trades = total_trades
    win_rate = (winning_trades / testable_trades * 100) if testable_trades > 0 else 0
    avg_layers_triggered = total_layers_triggered / total_trades if total_trades > 0 else 0
    avg_layers_breached = total_layers_breached / total_trades if total_trades > 0 else 0
    avg_recovery_pct = (total_recovery_amount / recovery_count) if recovery_count > 0 else 0
    roi = (total_actual_pnl / total_capital_deployed * 100) if total_capital_deployed > 0 else 0

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': testable_trades - winning_trades,
        'win_rate': round(win_rate, 2),
        'total_capital_deployed': round(total_capital_deployed, 2),
        'total_initial_credit': round(total_initial_credit, 2),
        'total_actual_pnl': round(total_actual_pnl, 2),
        'roi': round(roi, 2),
        'avg_layers_triggered': round(avg_layers_triggered, 2),
        'avg_layers_breached': round(avg_layers_breached, 2),
        'total_recovery_amount': round(total_recovery_amount, 2),
        'avg_recovery_pct': round(avg_recovery_pct, 2),
        'recovery_count': recovery_count,
        'layer_stats': layer_stats,
    }


def print_scale_in_statistics(
    aggregate_stats: Dict[str, Any],
    scale_in_results: List[Dict[str, Any]],
    scale_in_config: ScaleInConfig,
    comparison_results: Optional[List[Dict]] = None,
    summary_only: bool = False
):
    """Print comprehensive scale-in statistics."""
    print("\n" + "="*100)
    print("SCALE-IN ON BREACH STRATEGY ANALYSIS")
    print("="*100)

    print(f"\nCONFIGURATION:")
    print(f"  Total Capital: ${scale_in_config.total_capital:,.2f}")
    print(f"  Spread Width: ${scale_in_config.spread_width:.2f}")
    print(f"  Min Time Between Layers: {scale_in_config.min_time_between_layers_minutes} minutes")

    print(f"\n  PUT Layers:")
    for layer in scale_in_config.put_layers:
        print(f"    L{layer.level}: {layer.percent_beyond*100:.2f}% beyond | "
              f"{layer.capital_pct*100:.0f}% capital | Trigger: {layer.trigger}")

    print(f"\n  CALL Layers:")
    for layer in scale_in_config.call_layers:
        print(f"    L{layer.level}: {layer.percent_beyond*100:.2f}% beyond | "
              f"{layer.capital_pct*100:.0f}% capital | Trigger: {layer.trigger}")

    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Trades: {aggregate_stats['total_trades']}")
    print(f"  Winning Trades: {aggregate_stats['winning_trades']}")
    print(f"  Losing Trades: {aggregate_stats['losing_trades']}")
    print(f"  Win Rate: {aggregate_stats['win_rate']:.1f}%")
    print(f"  Total Capital Deployed: ${aggregate_stats['total_capital_deployed']:,.2f}")
    print(f"  Total Initial Credit: ${aggregate_stats['total_initial_credit']:,.2f}")

    pnl = aggregate_stats['total_actual_pnl']
    pnl_str = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    print(f"  Total P&L: {pnl_str}")
    print(f"  ROI: {aggregate_stats['roi']:+.2f}%")

    print(f"\nLAYER BREAKDOWN:")
    print(f"  Avg Layers Triggered: {aggregate_stats['avg_layers_triggered']:.2f}")
    print(f"  Avg Layers Breached: {aggregate_stats['avg_layers_breached']:.2f}")

    layer_stats = aggregate_stats.get('layer_stats', {})
    print(f"\n  {'Layer':<8} {'Triggered':<12} {'Breached':<12} {'P&L':<15}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*15}")

    for level in [1, 2, 3]:
        stats = layer_stats.get(level, {})
        triggered = stats.get('triggered', 0)
        breached = stats.get('breached', 0)
        pnl = stats.get('pnl', 0)
        pnl_str = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        print(f"  L{level:<7} {triggered:<12} {breached:<12} {pnl_str:<15}")

    if aggregate_stats.get('recovery_count', 0) > 0:
        print(f"\nRECOVERY ANALYSIS:")
        print(f"  Trades with Recovery: {aggregate_stats['recovery_count']}")
        print(f"  Total Recovery Amount: ${aggregate_stats['total_recovery_amount']:,.2f}")
        print(f"  Avg Recovery per Trade: ${aggregate_stats['total_recovery_amount']/aggregate_stats['recovery_count']:,.2f}")

    if comparison_results:
        print(f"\nCOMPARISON WITH SINGLE-ENTRY STRATEGY:")

        single_total_credits = 0
        single_total_gains = 0
        single_total_losses = 0
        single_winning = 0
        single_total = 0

        for result in comparison_results:
            backtest_result = result.get('backtest_successful')
            credit = result['best_spread'].get('total_credit') or result['best_spread'].get('net_credit_per_contract', 0)
            max_loss = result['best_spread'].get('total_max_loss') or result['best_spread'].get('max_loss_per_contract', 0)

            actual_pnl_per_share = result.get('actual_pnl_per_share')
            num_contracts = result['best_spread'].get('num_contracts', 1)

            if actual_pnl_per_share is not None and num_contracts:
                actual_pnl = actual_pnl_per_share * num_contracts * 100
            else:
                actual_pnl = credit if backtest_result else -max_loss

            single_total_credits += credit
            single_total += 1

            if backtest_result is True:
                single_winning += 1
                single_total_gains += actual_pnl if actual_pnl > 0 else credit
            elif backtest_result is False:
                single_total_losses += abs(actual_pnl) if actual_pnl < 0 else max_loss

        single_net_pnl = single_total_gains - single_total_losses
        single_win_rate = (single_winning / single_total * 100) if single_total > 0 else 0

        print(f"  {'Metric':<30} {'Single-Entry':<20} {'Scale-In':<20} {'Difference':<15}")
        print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*15}")
        print(f"  {'Total Trades':<30} {single_total:<20} {aggregate_stats['total_trades']:<20} {'-':<15}")
        print(f"  {'Win Rate':<30} {single_win_rate:.1f}%{'':<14} {aggregate_stats['win_rate']:.1f}%{'':<14} {aggregate_stats['win_rate'] - single_win_rate:+.1f}%")

        single_pnl_str = f"${single_net_pnl:,.2f}" if single_net_pnl >= 0 else f"-${abs(single_net_pnl):,.2f}"
        scale_pnl_str = f"${aggregate_stats['total_actual_pnl']:,.2f}" if aggregate_stats['total_actual_pnl'] >= 0 else f"-${abs(aggregate_stats['total_actual_pnl']):,.2f}"
        diff_pnl = aggregate_stats['total_actual_pnl'] - single_net_pnl
        diff_pnl_str = f"${diff_pnl:+,.2f}"

        print(f"  {'Net P&L':<30} {single_pnl_str:<20} {scale_pnl_str:<20} {diff_pnl_str:<15}")

        if single_total_losses > 0:
            single_total_str = f"${single_total_losses:,.2f}"
            scale_losses = aggregate_stats.get('total_actual_pnl', 0)
            if scale_losses < 0:
                scale_total_losses = abs(scale_losses)
            else:
                scale_total_losses = 0
            scale_total_str = f"${scale_total_losses:,.2f}"

            recovery = single_total_losses - scale_total_losses
            recovery_pct = (recovery / single_total_losses * 100) if single_total_losses > 0 else 0

            print(f"  {'Total Losses':<30} {single_total_str:<20} {scale_total_str:<20} {recovery_pct:.1f}% saved")

    if not summary_only and scale_in_results:
        print(f"\nINDIVIDUAL TRADE DETAILS:")
        print("-"*100)

        for i, result in enumerate(scale_in_results[:20], 1):
            summary = result['summary']
            print(f"\n{i}. Date: {summary['trading_date']}, Type: {summary['option_type'].upper()}")
            print(f"   Prev Close: ${summary['prev_close']:,.2f}")
            print(f"   Layers Triggered: {summary['num_layers_triggered']}, Breached: {summary['num_layers_breached']}")
            print(f"   Capital: ${summary['total_capital_deployed']:,.2f}, Credit: ${summary['total_initial_credit']:,.2f}")

            pnl = summary.get('total_actual_pnl')
            if pnl is not None:
                pnl_str = f"${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
                print(f"   P&L: {pnl_str}")

            if 'recovery_vs_single' in summary:
                recovery = summary['recovery_vs_single']
                print(f"   Recovery: ${recovery['recovery_amount']:,.2f} ({recovery['recovery_pct']:.1f}%)")

        if len(scale_in_results) > 20:
            print(f"\n... and {len(scale_in_results) - 20} more trades")

    print("\n" + "="*100)


# ============================================================================
# STRATEGY HELPER
# ============================================================================


def _execute_strategy_on_results(strategy, results, output_tz, logger, args, min_premium_diff=None):
    """Run strategy framework on analysis results.

    Called from both main() and the continuous runner to execute a strategy
    against a set of interval analysis results.
    """
    print(f"\n{'='*100}")
    print(f"Running {strategy.name.upper().replace('_', ' ')} Strategy via Strategy Framework...")
    print(f"{'='*100}")

    strategy_results_by_date = defaultdict(lambda: {'put': [], 'call': []})
    for result in results:
        timestamp = result['timestamp']
        if hasattr(timestamp, 'date'):
            trading_date = timestamp.date()
        else:
            trading_date = pd.to_datetime(timestamp).date()
        opt_type = result.get('option_type', 'unknown').lower()
        if opt_type in ['put', 'call']:
            strategy_results_by_date[trading_date][opt_type].append(result)

    strategy_results = []
    for trading_date, type_results in sorted(strategy_results_by_date.items()):
        for opt_type in ['put', 'call']:
            day_results = type_results[opt_type]
            if not day_results:
                continue

            first_result = day_results[0]
            prev_close = first_result.get('prev_close')
            current_close = first_result.get('current_close')

            if prev_close is None or current_close is None:
                logger.debug(f"Skipping strategy {trading_date} {opt_type}: missing price data")
                continue

            trading_datetime = datetime.combine(trading_date, datetime.min.time())
            if output_tz:
                try:
                    trading_datetime = output_tz.localize(trading_datetime)
                except AttributeError:
                    trading_datetime = trading_datetime.replace(tzinfo=output_tz)

            try:
                positions = strategy.select_entries(
                    day_results=day_results,
                    prev_close=prev_close,
                    option_type=opt_type,
                    trading_date=trading_datetime,
                    min_premium_diff=min_premium_diff,
                    max_credit_width_ratio=args.max_credit_width_ratio,
                    min_contract_price=args.min_contract_price,
                    max_strike_distance_pct=args.max_strike_distance_pct,
                )

                # Compute single-entry P&L for comparison
                single_entry_pnl = None
                for r in day_results:
                    pnl = r.get('actual_pnl_per_share')
                    bs = r.get('best_spread', {})
                    n = bs.get('num_contracts') or 0
                    if pnl is not None and n > 0:
                        single_entry_pnl = pnl * n * 100
                        break

                strategy_result = strategy.calculate_pnl(
                    positions=positions,
                    close_price=current_close,
                    option_type=opt_type,
                    trading_date=trading_datetime,
                    single_entry_pnl=single_entry_pnl,
                )
                strategy_results.append(strategy_result)
            except Exception as e:
                logger.warning(f"Error running strategy for {trading_date} {opt_type}: {e}")
                continue

    if strategy_results:
        total_pnl = sum(sr.total_pnl or 0 for sr in strategy_results)
        total_credit = sum(sr.total_credit for sr in strategy_results)
        total_max_loss = sum(sr.total_max_loss for sr in strategy_results)
        winning = sum(1 for sr in strategy_results if sr.total_pnl and sr.total_pnl > 0)
        total = len(strategy_results)
        win_rate = (winning / total * 100) if total > 0 else 0
        roi = (total_pnl / total_max_loss * 100) if total_max_loss > 0 else 0

        print(f"\nSTRATEGY RESULTS ({strategy.name}):")
        print(f"  Total Trades: {total}")
        print(f"  Winning: {winning}, Losing: {total - winning}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total Credit: ${total_credit:,.2f}")
        print(f"  Total Max Loss: ${total_max_loss:,.2f}")
        pnl_str = f"${total_pnl:,.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):,.2f}"
        print(f"  Net P&L: {pnl_str}")
        print(f"  ROI: {roi:+.2f}%")
        if strategy.config.feature_flags:
            print(f"  Feature Flags: {strategy.config.feature_flags}")

        # Per-day, per-window position detail
        print(f"\n  {'─'*96}")
        print(f"  {'Date':<12} {'Side':<5} {'Window':<8} {'Tier':<5} {'Contracts':>9} "
              f"{'Short':>12} {'Long':>12} {'Width':>6} {'Credit':>12} {'ROI':>7} {'P&L':>12}")
        print(f"  {'─'*96}")
        for sr in strategy_results:
            date_str = sr.trading_date.strftime('%Y-%m-%d') if sr.trading_date else '?'
            has_positions = False
            for pos_info in sr.positions:
                ts = pos_info.get('trade_state')
                if ts is None:
                    continue
                for wd in ts.window_deployments:
                    for pos in wd.deployed_positions:
                        has_positions = True
                        pnl_val = pos.actual_pnl_total or 0
                        pnl_str = f"${pnl_val:>+10,.2f}"
                        print(f"  {date_str:<12} {pos.option_type:<5} {wd.window_label:<8} "
                              f"T{pos.tier_level:<4} {pos.num_contracts:>9} "
                              f"${pos.short_strike:>10,.2f} ${pos.long_strike:>10,.2f} "
                              f"{pos.spread_width:>5.0f} ${pos.initial_credit_total:>10,.2f} "
                              f"{pos.roi*100:>6.2f}% {pnl_str}")
            if not has_positions:
                print(f"  {date_str:<12} {sr.option_type:<5} {'—':^8} {'—':^5} {'—':>9} "
                      f"{'—':>12} {'—':>12} {'—':>6} {'—':>12} {'—':>7} {'$0.00':>12}")
        print(f"  {'─'*96}")
    else:
        print(f"No {strategy.name} trades could be analyzed (missing price data)")

    print(f"\n{'='*100}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    args = parse_args()

    # Mode dispatch — non-credit-spread modes bypass all existing validation
    if args.mode == 'price-movements':
        return run_price_movement_analysis(args)
    elif args.mode == 'max-move':
        return run_max_move_analysis(args)
    elif args.mode == 'risk-gradient':
        logger = get_logger("ndx_risk_gradient", level=getattr(args, 'log_level', 'INFO'))
        return await run_risk_gradient_analysis(args, logger)
    elif args.mode == 'dte-comparison':
        return await run_dte_comparison(args)

    # Handle --clear-cache
    if args.clear_cache:
        clear_cache(args.cache_dir)
        return 0

    # Handle --grid-config (grid search mode)
    if args.grid_config:
        return await run_grid_search(args)

    # Validate --percent-beyond is required in normal mode
    if not args.percent_beyond:
        print("Error: --percent-beyond is required (unless using --grid-config)")
        return 1

    # Validate that either csv_path or csv_dir is provided
    if not args.csv_path and not args.csv_dir:
        print("Error: Either --csv-path or --csv-dir must be provided")
        return 1

    if args.csv_path and args.csv_dir:
        print("Error: Cannot use both --csv-path and --csv-dir. Use one or the other.")
        return 1

    if args.csv_dir and not args.underlying_ticker:
        print("Error: --csv-dir requires --ticker or --underlying-ticker to be specified")
        return 1

    if args.end_date and not args.start_date and not args.csv_dir:
        print("Error: --end-date requires --start-date or --csv-dir")
        return 1

    if args.risk_cap is None and args.max_spread_width is None:
        print("Error: Either --risk-cap or --max-spread-width must be provided")
        return 1

    if args.best_only and not args.most_recent:
        print("Error: --best-only requires --most-recent to be enabled")
        return 1

    if args.curr_price and args.continuous is None:
        print("Error: --curr-price requires --continuous mode to be enabled")
        return 1

    if args.use_market_hours and args.continuous is None:
        print("Error: --use-market-hours requires --continuous mode to be enabled")
        return 1

    if args.run_once_before_wait and args.continuous is None:
        print("Error: --run-once-before-wait requires --continuous mode to be enabled")
        return 1

    if args.scale_in_enabled and not args.scale_in_config:
        print("Error: --scale-in-enabled requires --scale-in-config to specify a configuration file")
        return 1

    # Load scale-in configuration if provided
    scale_in_config = None
    if args.scale_in_config:
        try:
            scale_in_config = load_scale_in_config(args.scale_in_config)
            if args.scale_in_enabled:
                print(f"Scale-in strategy enabled with config: {args.scale_in_config}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    if args.tiered_enabled and not args.tiered_config:
        print("Error: --tiered-enabled requires --tiered-config to specify a configuration file")
        return 1

    # Load tiered investment configuration if provided
    tiered_config = None
    if args.tiered_config:
        try:
            tiered_config = load_tiered_config(args.tiered_config)
            if args.tiered_enabled:
                print(f"Tiered investment strategy enabled with config: {args.tiered_config}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Instantiate strategy from --strategy / --strategy-config if provided
    strategy = None
    if args.strategy:
        import json as _json
        strategy_config_dict = {}
        if args.strategy_config:
            try:
                with open(args.strategy_config, 'r') as _f:
                    strategy_config_dict = _json.load(_f)
            except (IOError, ValueError) as e:
                print(f"Error loading strategy config '{args.strategy_config}': {e}")
                return 1

        # Backward compat: if --strategy scale_in but no config_file in JSON, use --scale-in-config
        if args.strategy == 'scale_in' and 'config_file' not in strategy_config_dict and args.scale_in_config:
            strategy_config_dict['config_file'] = args.scale_in_config
        # Backward compat: if --strategy tiered but no config_file in JSON, use --tiered-config
        if args.strategy == 'tiered' and 'config_file' not in strategy_config_dict and args.tiered_config:
            strategy_config_dict['config_file'] = args.tiered_config

        try:
            strategy = StrategyRegistry.create(args.strategy, strategy_config_dict, logger=None)
            strategy.validate_config()
            print(f"Strategy: {strategy.name} (feature flags: {strategy.config.feature_flags or 'none'})")
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    try:
        output_tz = resolve_timezone(args.output_timezone)
    except Exception as e:
        print(f"Error: Invalid --output-timezone '{args.output_timezone}': {e}")
        return 1

    logger = get_logger("analyze_credit_spread_intervals", level=args.log_level)
    if strategy:
        strategy.logger = logger

    # Parse percent-beyond value
    try:
        percent_beyond = parse_percent_beyond(args.percent_beyond)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Parse max-spread-width value
    try:
        max_spread_width = parse_max_spread_width(args.max_spread_width)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Parse min-premium-diff value (if provided)
    min_premium_diff = None
    if args.min_premium_diff:
        try:
            min_premium_diff = parse_min_premium_diff(args.min_premium_diff)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    # Build delta filter config from CLI arguments
    delta_filter_config = None
    delta_filtering_active = (
        args.max_short_delta is not None or
        args.min_short_delta is not None or
        args.max_long_delta is not None or
        args.min_long_delta is not None or
        args.delta_range is not None or
        args.require_delta
    )
    if delta_filtering_active:
        min_short_delta = args.min_short_delta
        max_short_delta = args.max_short_delta
        if args.delta_range:
            parsed_min, parsed_max = parse_delta_range(args.delta_range)
            if parsed_min is not None:
                min_short_delta = parsed_min
            if parsed_max is not None:
                max_short_delta = parsed_max

        delta_filter_config = DeltaFilterConfig(
            max_short_delta=max_short_delta,
            min_short_delta=min_short_delta,
            max_long_delta=args.max_long_delta,
            min_long_delta=args.min_long_delta,
            require_delta=args.require_delta,
            default_iv=args.delta_default_iv,
            use_vix1d=args.use_vix1d,
            vix1d_dir=args.vix1d_dir,
        )
        logger.info(format_delta_filter_info(delta_filter_config))

    # Build close predictor gate config
    cp_gate = None
    if args.close_predictor:
        if not args.underlying_ticker:
            print("Error: --close-predictor requires --ticker or --underlying-ticker to be specified")
            return 1
        try:
            buf_pts, buf_pct = parse_close_predictor_buffer(args.close_predictor_buffer)
        except ValueError as e:
            print(f"Error: Invalid --close-predictor-buffer value: {e}")
            return 1
        cp_gate_config = ClosePredictorGateConfig(
            enabled=True,
            band_level=args.close_predictor_level,
            buffer_points=buf_pts,
            buffer_pct=buf_pct,
            mode=args.close_predictor_mode,
            lookback=args.close_predictor_lookback,
        )
        cp_gate = ClosePredictorGate(cp_gate_config, args.underlying_ticker, logger)
        logger.info(
            f"Close predictor gate enabled: level={cp_gate_config.band_level}, "
            f"buffer_points={buf_pts}, buffer_pct={buf_pct}, mode={cp_gate_config.mode}"
        )

    # Determine CSV file paths
    if args.csv_dir:
        ticker = args.underlying_ticker
        if not ticker:
            print("Error: --ticker or --underlying-ticker is required when using --csv-dir")
            return 1

        csv_paths = find_csv_files_in_dir(
            args.csv_dir,
            ticker,
            args.start_date,
            args.end_date,
            logger
        )

        if not csv_paths:
            print(f"Error: No CSV files found in {args.csv_dir}/{ticker.upper()}/ matching the criteria")
            return 1

        csv_paths = [str(p) for p in csv_paths]
        logger.info(f"Found {len(csv_paths)} CSV file(s) from --csv-dir")
    else:
        csv_paths = args.csv_path if isinstance(args.csv_path, list) else [args.csv_path]
        logger.info(f"Reading {len(csv_paths)} CSV file(s) from --csv-path")

    # Determine if we should use multiprocessing
    num_processes = args.processes
    use_multiprocessing = len(csv_paths) > 1 and num_processes != 1

    if num_processes == 0:
        num_processes = multiprocessing.cpu_count()
        logger.info(f"Auto-detected {num_processes} CPUs")

    # Determine which option types to analyze
    option_types_to_analyze = []
    if args.option_type == "both":
        option_types_to_analyze = ["call", "put"]
    else:
        option_types_to_analyze = [args.option_type]

    # Handle continuous mode
    if args.continuous is not None:
        if args.use_market_hours:
            now_utc = datetime.now(timezone.utc)
            is_market_open = is_market_hours()
            seconds_to_open, _ = compute_market_transition_times(now_utc, args.output_timezone)

            if not is_market_open and seconds_to_open is not None:
                if args.run_once_before_wait:
                    print(f"Market is closed. Running once immediately before waiting for market open...")
                    await _run_single_analysis_iteration(
                        args, csv_paths, percent_beyond, max_spread_width,
                        option_types_to_analyze, output_tz, logger, min_premium_diff, delta_filter_config,
                        close_predictor_gate=cp_gate
                    )
                    hours_to_wait = seconds_to_open / 3600
                    print(f"One-time run completed. Waiting {hours_to_wait:.2f} hours ({seconds_to_open:.0f} seconds) until market opens...")
                    await asyncio.sleep(seconds_to_open)
                    now_utc = datetime.now(timezone.utc)
                    is_market_open = is_market_hours()
                    if is_market_open:
                        print("Market is now open. Proceeding with normal operation...")
                    else:
                        print("Warning: Market is still not open after waiting. Proceeding anyway...")
                else:
                    pre_open_buffer = 300
                    if seconds_to_open > pre_open_buffer:
                        wait_until_buffer = seconds_to_open - pre_open_buffer
                        hours_to_wait = wait_until_buffer / 3600
                        print(
                            f"Market is closed. Waiting {hours_to_wait:.2f} hours "
                            f"({wait_until_buffer:.0f} seconds) so we wake up 5 minutes before market open..."
                        )
                        await asyncio.sleep(wait_until_buffer)
                        print("Pre-market wake-up reached. Starting analysis 5 minutes before market open...")
                    else:
                        print(
                            f"Market opens in {seconds_to_open/60:.1f} minutes. "
                            "Starting analysis now so it is running before the open..."
                        )
                    now_utc = datetime.now(timezone.utc)
                    is_market_open = is_market_hours()
                    if is_market_open:
                        print("Market is now open. Starting analysis...")
                    else:
                        print("Market still closed, beginning pre-open analysis cadence...")

        await run_continuous_analysis(
            args, csv_paths, percent_beyond, max_spread_width,
            option_types_to_analyze, output_tz, logger, min_premium_diff, delta_filter_config,
            close_predictor_gate=cp_gate,
            strategy=strategy,
        )
        return 0

    # Process CSV files (normal mode, not continuous)
    if use_multiprocessing:
        logger.info(f"Processing {len(csv_paths)} files using {num_processes} parallel processes")

        process_args = []
        for csv_path in csv_paths:
            args_tuple = (
                csv_path,
                option_types_to_analyze,
                percent_beyond,
                args.risk_cap,
                args.min_spread_width,
                max_spread_width,
                args.use_mid_price,
                args.min_contract_price,
                args.underlying_ticker,
                args.db_path,
                args.no_cache,
                args.log_level,
                args.max_credit_width_ratio,
                args.max_strike_distance_pct,
                args.curr_price and args.continuous is not None,
                args.max_trading_hour,
                args.min_trading_hour,
                args.profit_target_pct,
                args.most_recent,
                output_tz,
                args.force_close_hour,
                args.cache_dir,
                args.no_data_cache,
                min_premium_diff,
                args.rate_limit_max,
                args.rate_limit_window,
                getattr(args, 'rate_limit_blocks', None),
            )
            process_args.append(args_tuple)

        with multiprocessing.Pool(processes=num_processes) as pool:
            results_list = pool.map(process_single_csv_sync, process_args)

        results = []
        for file_results in results_list:
            results.extend(file_results)

        logger.info(f"Parallel processing complete. Total results: {len(results)}")

        # Apply close predictor gate filter (before capital filtering)
        if cp_gate is not None and cp_gate.config.enabled:
            results = cp_gate.filter_results(results)

        if args.max_live_capital is not None:
            original_count = len(results)
            results = filter_results_by_capital_limit(
                results,
                args.max_live_capital,
                output_tz,
                logger
            )
            logger.info(
                f"Capital limit filter: {original_count} -> {len(results)} positions "
                f"(max ${args.max_live_capital:,.2f} per day)"
            )

            daily_capital_usage = {}
            for result in results:
                position_capital, calendar_date = calculate_position_capital(result, output_tz)
                daily_capital_usage[calendar_date] = daily_capital_usage.get(calendar_date, 0.0) + position_capital

            if daily_capital_usage:
                logger.info("Final daily capital usage:")
                for date, capital in sorted(daily_capital_usage.items()):
                    logger.info(f"  {date}: ${capital:,.2f} / ${args.max_live_capital:,.2f} ({(capital/args.max_live_capital*100):.1f}%)")

        skip_normal_processing = True
    else:
        logger.info(f"Processing {len(csv_paths)} file(s) sequentially")

        try:
            df = load_data_cached(
                csv_paths,
                cache_dir=args.cache_dir,
                no_cache=args.no_data_cache,
                logger=logger
            )
        except ValueError as e:
            logger.error(str(e))
            return 1
        except Exception as e:
            logger.error(f"Failed to load CSV data: {e}")
            return 1

        skip_normal_processing = False

    # Normal processing (when not using multiprocessing)
    if not skip_normal_processing:

        logger.info("Initializing database connection...")
        if args.db_path:
            db_config = args.db_path
        else:
            db_config = os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '')
        db = StockQuestDB(
            db_config,
            enable_cache=not args.no_cache,
            logger=logger
        )

        try:
            intervals_grouped = df.groupby('interval')
            total_intervals_count = len(intervals_grouped)

            if args.most_recent:
                max_interval = df['interval'].max()
                max_interval_df = df[df['interval'] == max_interval]
                intervals_to_process = [(max_interval, max_interval_df)]
                logger.info(f"Analyzing most recent interval only: {max_interval}")
            else:
                intervals_to_process = intervals_grouped
                logger.info(f"Analyzing {total_intervals_count} intervals...")

            results = []

            time_block_limiter = None
            sliding_limiter = None

            if hasattr(args, 'rate_limit_blocks') and args.rate_limit_blocks:
                time_block_limiter = TimeBlockRateLimiter.from_string(args.rate_limit_blocks, logger=logger)
                logger.info(f"Time-block rate limiting enabled: {args.rate_limit_blocks}")
            elif args.rate_limit_max > 0 and args.rate_limit_window > 0:
                sliding_limiter = SlidingWindowRateLimiter(
                    max_transactions=args.rate_limit_max,
                    window_seconds=args.rate_limit_window,
                    logger=logger
                )
                logger.info(f"Sliding window rate limiting enabled: {args.rate_limit_max} transactions per {args.rate_limit_window}s")

            for interval_time, interval_df in intervals_to_process:
                for opt_type in option_types_to_analyze:
                    if time_block_limiter:
                        await time_block_limiter.acquire()
                    elif sliding_limiter:
                        await sliding_limiter.acquire()
                    use_current_price = args.curr_price and args.continuous is not None
                    result = await analyze_interval(
                        db,
                        interval_df,
                        opt_type,
                        percent_beyond,
                        args.risk_cap,
                        args.min_spread_width,
                        max_spread_width,
                        args.use_mid_price,
                        args.min_contract_price,
                        args.underlying_ticker,
                        logger,
                        args.max_credit_width_ratio,
                        args.max_strike_distance_pct,
                        use_current_price,
                        args.max_trading_hour,
                        args.min_trading_hour,
                        args.profit_target_pct,
                        output_tz,
                        args.force_close_hour,
                        min_premium_diff,
                        None,  # dynamic_width_config
                        delta_filter_config,
                    )
                    if result:
                        results.append(result)

            # Apply close predictor gate filter (before capital filtering)
            if cp_gate is not None and cp_gate.config.enabled:
                results = cp_gate.filter_results(results)

            if args.max_live_capital is not None:
                original_count = len(results)
                results = filter_results_by_capital_limit(
                    results,
                    args.max_live_capital,
                    output_tz,
                    logger
                )
                logger.info(
                    f"Capital limit filter: {original_count} -> {len(results)} positions "
                    f"(max ${args.max_live_capital:,.2f} per day)"
                )

                daily_capital_usage = {}
                for result in results:
                    position_capital, calendar_date = calculate_position_capital(result, output_tz)
                    daily_capital_usage[calendar_date] = daily_capital_usage.get(calendar_date, 0.0) + position_capital

                if daily_capital_usage:
                    logger.info("Final daily capital usage:")
                    for date, capital in sorted(daily_capital_usage.items()):
                        logger.info(f"  {date}: ${capital:,.2f} / ${args.max_live_capital:,.2f} ({(capital/args.max_live_capital*100):.1f}%)")

        finally:
            await db.close()

    # Post-processing (common for both multiprocessing and sequential)
    if skip_normal_processing:
        total_intervals_count = len(results)

    # Apply top-N filtering
    original_results_count = len(results)
    if args.top_n and results:
        results = filter_top_n_per_day(results, args.top_n)
        logger.info(f"Applied top-{args.top_n} per day filter: {original_results_count} -> {len(results)} results")

    # Handle --most-recent mode
    if args.most_recent:
        if results:
            max_timestamp = max(result['timestamp'] for result in results)
            most_recent_results = []
            call_results = [r for r in results if r['timestamp'] == max_timestamp and r.get('option_type', '').lower() == 'call']
            put_results = [r for r in results if r['timestamp'] == max_timestamp and r.get('option_type', '').lower() == 'put']

            best_call = None
            best_put = None

            if call_results:
                best_call = max(call_results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0))

            if put_results:
                best_put = max(put_results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0))

            if args.best_only:
                if best_call and best_put:
                    call_credit = best_call['best_spread'].get('total_credit') or best_call['best_spread'].get('net_credit_per_contract', 0)
                    put_credit = best_put['best_spread'].get('total_credit') or best_put['best_spread'].get('net_credit_per_contract', 0)
                    if call_credit > put_credit:
                        most_recent_results = [best_call]
                    else:
                        most_recent_results = [best_put]
                elif best_call:
                    most_recent_results = [best_call]
                elif best_put:
                    most_recent_results = [best_put]
            else:
                if best_call:
                    most_recent_results.append(best_call)
                if best_put:
                    most_recent_results.append(best_put)

            results = most_recent_results

            if args.best_only and args.continuous is not None:
                if results:
                    print(format_best_current_option(results[0], output_tz, args.use_mid_price))
                else:
                    most_recent_ts = None
                    try:
                        if df is not None and len(df) > 0:
                            most_recent_ts = df['timestamp'].max()
                    except (NameError, UnboundLocalError):
                        pass
                    if most_recent_ts:
                        max_timestamp_str = format_timestamp(most_recent_ts, output_tz)
                        print(f"NO RESULTS: No valid spreads found at most recent timestamp {max_timestamp_str} that meet the criteria.")
                    else:
                        print("NO RESULTS: No valid spreads found.")
                    return 0
        else:
            most_recent_ts = None
            try:
                if df is not None and len(df) > 0:
                    most_recent_ts = df['timestamp'].max()
            except (NameError, UnboundLocalError):
                pass
            if most_recent_ts:
                most_recent_str = format_timestamp(most_recent_ts, output_tz)
                if args.best_only and args.continuous is not None:
                    print(f"NO RESULTS: No valid spreads found at most recent timestamp {most_recent_str} that meet the criteria.")
                else:
                    print(f"NO RESULTS: No valid spreads found. Most recent data timestamp: {most_recent_str}")
            else:
                print("NO RESULTS: No valid spreads found.")
            return 0

    # Print results
    if args.summary or args.summary_only:
        if results:
            sorted_results = sorted(results, key=lambda x: x['timestamp'])

            # Print individual summary lines
            if args.summary and not args.summary_only and not (args.best_only and args.continuous is not None):
                for result in sorted_results:
                    print(format_summary_line(result, output_tz, args.use_mid_price))

            # Print final one-line summary
            summary_parts = build_summary_parts(
                results, output_tz, args.option_type,
                top_n=args.top_n, use_mid_price=args.use_mid_price
            )
            if summary_parts:
                print(f"SUMMARY: {' | '.join(summary_parts)}")
        else:
            print("No valid credit spreads found matching the criteria.")
    else:
        # Detailed view
        option_type_display = args.option_type.upper() if args.option_type != "both" else "CALL & PUT"
        print("\n" + "="*100)
        print(f"CREDIT SPREAD ANALYSIS - {option_type_display} OPTIONS")
        print("="*100)
        if args.csv_dir:
            print(f"CSV Directory: {args.csv_dir}")
            print(f"CSV Files: {len(csv_paths)} file(s)")
            if args.start_date or args.end_date:
                date_range = f"{args.start_date or 'beginning'} to {args.end_date or 'today'}"
                print(f"Date Range: {date_range}")
        else:
            print(f"CSV File(s): {args.csv_path if isinstance(args.csv_path, list) else args.csv_path}")
        print(f"Option Type: {args.option_type}")
        print(f"Underlying Ticker: {args.underlying_ticker or 'Auto-detected from CSV'}")
        put_pct, call_pct = percent_beyond
        if put_pct == call_pct:
            print(f"Percent Beyond Previous Close: {put_pct * 100:.2f}%")
        else:
            print(f"Percent Beyond Previous Close: PUT {put_pct * 100:.2f}% / CALL {call_pct * 100:.2f}%")
        print(f"Output Timezone: {args.output_timezone}")
        if args.risk_cap is not None:
            print(f"Risk Cap: ${args.risk_cap:.2f}")
        put_max_width, call_max_width = max_spread_width
        if put_max_width == call_max_width:
            print(f"Max Spread Width: ${put_max_width:.2f}")
        else:
            print(f"Max Spread Width: PUT ${put_max_width:.2f} / CALL ${call_max_width:.2f}")
        print(f"Min Contract Price: ${args.min_contract_price:.2f}")
        try:
            tz_display = output_tz.tzname(datetime.now())
        except:
            tz_display = args.output_timezone
        if args.min_trading_hour is not None:
            print(f"Min Trading Hour: {args.min_trading_hour}:00 {tz_display}")
        print(f"Max Trading Hour: {args.max_trading_hour}:00 {tz_display}")
        if args.max_live_capital is not None:
            print(f"Max Live Capital: ${args.max_live_capital:,.2f} per day (max loss exposure)")
        if args.force_close_hour is not None:
            print(f"Force Close Hour: {args.force_close_hour}:00 {tz_display} (all positions closed, P&L calculated)")
        if args.profit_target_pct is not None:
            print(f"Profit Target: {args.profit_target_pct * 100:.0f}% of max credit")
        if use_multiprocessing:
            print(f"Parallel Processing: {num_processes} processes")
        print(f"Total Intervals Analyzed: {total_intervals_count}")
        if args.top_n:
            print(f"Intervals with Valid Spreads: {len(results)} (Top-{args.top_n} per day from {original_results_count} total)")
        else:
            print(f"Intervals with Valid Spreads: {len(results)}")
        print("="*100)

        if results:
                overall_best = max(results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', x['best_spread']['net_credit'] * 100))

                print(f"\nOVERALL BEST SPREAD:")
                print(f"  Timestamp: {format_timestamp(overall_best['timestamp'], output_tz)}")
                print(f"  Underlying: {overall_best['underlying']}")
                print(f"  Option Type: {overall_best.get('option_type', 'UNKNOWN').upper()}")
                print(f"  Previous Close: ${overall_best['prev_close']:.2f} (from {overall_best['prev_close_date']})")
                if overall_best.get('current_close') is not None:
                    print(f"  Current Day Close: ${overall_best['current_close']:.2f} (from {overall_best.get('current_close_date', 'N/A')})")
                    if overall_best.get('price_diff_pct') is not None:
                        print(f"  Price Change: {overall_best['price_diff_pct']:+.2f}%")
                print(f"  Target Price: ${overall_best['target_price']:.2f}")
                print(f"  Short Strike: ${overall_best['best_spread']['short_strike']:.2f}")
                print(f"  Long Strike: ${overall_best['best_spread']['long_strike']:.2f}")
                print(f"  Short Premium: ${overall_best['best_spread']['short_price']:.2f}")
                print(f"  Long Premium: ${overall_best['best_spread']['long_price']:.2f}")
                print(f"  Spread Width: ${overall_best['best_spread']['width']:.2f} (per share)")
                print(f"  Net Credit (per share): ${overall_best['best_spread']['net_credit']:.2f}")
                print(f"  Net Credit (per contract): ${overall_best['best_spread']['net_credit_per_contract']:.2f}")
                print(f"  Max Loss (per share): ${overall_best['best_spread']['max_loss']:.2f}")
                print(f"  Max Loss (per contract): ${overall_best['best_spread']['max_loss_per_contract']:.2f}")
                print(f"  Risk/Reward: {overall_best['best_spread']['net_credit_per_contract'] / overall_best['best_spread']['max_loss_per_contract']:.2f}")

                if overall_best['best_spread']['short_delta'] is not None:
                    print(f"  Short Delta: {overall_best['best_spread']['short_delta']:.4f}")
                if overall_best['best_spread']['long_delta'] is not None:
                    print(f"  Long Delta: {overall_best['best_spread']['long_delta']:.4f}")
                if overall_best['best_spread']['net_delta'] is not None:
                    print(f"  Net Delta: {overall_best['best_spread']['net_delta']:.4f}")

                if overall_best['best_spread']['num_contracts'] is not None:
                    print(f"  Number of Contracts: {overall_best['best_spread']['num_contracts']}")
                    print(f"  Total Credit: ${overall_best['best_spread']['total_credit']:.2f}")
                    print(f"  Total Max Loss: ${overall_best['best_spread']['total_max_loss']:.2f}")

                backtest_result = overall_best.get('backtest_successful')
                profit_target_hit = overall_best.get('profit_target_hit')
                actual_pnl_per_share = overall_best.get('actual_pnl_per_share')
                close_price_used = overall_best.get('close_price_used')
                close_time_used = overall_best.get('close_time_used')

                if backtest_result is True:
                    if profit_target_hit is True:
                        print(f"  Backtest: \u2713 SUCCESS (Profit target hit early)")
                    else:
                        print(f"  Backtest: \u2713 SUCCESS (EOD close did not breach spread)")
                elif backtest_result is False:
                    print(f"  Backtest: \u2717 FAILURE (EOD close breached spread)")

                if actual_pnl_per_share is not None:
                    num_contracts = overall_best['best_spread'].get('num_contracts', 1)
                    if num_contracts:
                        total_pnl = actual_pnl_per_share * num_contracts * 100
                        print(f"  Actual P&L (per share): ${actual_pnl_per_share:+.2f}")
                        print(f"  Actual P&L (total): ${total_pnl:+.2f}")
                    else:
                        print(f"  Actual P&L (per share): ${actual_pnl_per_share:+.2f}")

                    if close_price_used is not None:
                        close_price_display = f"${close_price_used:.2f}"
                        if close_time_used is not None:
                            close_time_display = format_timestamp(close_time_used, output_tz)
                            print(f"  Close Price Used: {close_price_display} at {close_time_display}")
                        else:
                            print(f"  Close Price Used: {close_price_display}")

                print(f"\nALL INTERVALS WITH VALID SPREADS:")
                print("-"*100)
                for i, result in enumerate(sorted(results, key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', x['best_spread']['net_credit'] * 100), reverse=True), 1):
                    print(f"\n{i}. Interval: {format_timestamp(result['timestamp'], output_tz)}")
                    print(f"   Underlying: {result['underlying']}, Type: {result.get('option_type', 'UNKNOWN').upper()}, Prev Close: ${result['prev_close']:.2f} (from {result['prev_close_date']})")
                    if result.get('current_close') is not None:
                        print(f"   Current Day Close: ${result['current_close']:.2f} (from {result.get('current_close_date', 'N/A')})", end="")
                        if result.get('price_diff_pct') is not None:
                            print(f", Price Change: {result['price_diff_pct']:+.2f}%")
                        else:
                            print()
                    print(f"   Best Spread: ${result['best_spread']['short_strike']:.2f} / ${result['best_spread']['long_strike']:.2f}")
                    print(f"   Short Premium: ${result['best_spread']['short_price']:.2f}, Long Premium: ${result['best_spread']['long_price']:.2f}")
                    print(f"   Credit (per contract): ${result['best_spread']['net_credit_per_contract']:.2f}, Max Loss (per contract): ${result['best_spread']['max_loss_per_contract']:.2f}")

                    delta_info = []
                    if result['best_spread']['short_delta'] is not None:
                        delta_info.append(f"Short \u0394: {result['best_spread']['short_delta']:.4f}")
                    if result['best_spread']['long_delta'] is not None:
                        delta_info.append(f"Long \u0394: {result['best_spread']['long_delta']:.4f}")
                    if result['best_spread']['net_delta'] is not None:
                        delta_info.append(f"Net \u0394: {result['best_spread']['net_delta']:.4f}")
                    if delta_info:
                        print(f"   {' | '.join(delta_info)}")

                    if result['best_spread']['num_contracts'] is not None:
                        print(f"   Contracts: {result['best_spread']['num_contracts']}, Total Credit: ${result['best_spread']['total_credit']:.2f}, Total Max Loss: ${result['best_spread']['total_max_loss']:.2f}")

                    backtest_result = result.get('backtest_successful')
                    profit_target_hit = result.get('profit_target_hit')
                    actual_pnl_per_share = result.get('actual_pnl_per_share')

                    if backtest_result is True:
                        if profit_target_hit is True:
                            print(f"   Backtest: \u2713 SUCCESS (Profit target hit early)")
                        else:
                            print(f"   Backtest: \u2713 SUCCESS (EOD close did not breach spread)")
                    elif backtest_result is False:
                        print(f"   Backtest: \u2717 FAILURE (EOD close breached spread)")

                    if actual_pnl_per_share is not None:
                        num_contracts = result['best_spread'].get('num_contracts', 1)
                        if num_contracts:
                            total_pnl = actual_pnl_per_share * num_contracts * 100
                            print(f"   Actual P&L: ${actual_pnl_per_share:+.2f} per share, ${total_pnl:+.2f} total")
                        else:
                            print(f"   Actual P&L: ${actual_pnl_per_share:+.2f} per share")

                    print(f"   Total Valid Spreads: {result['total_spreads']}")
        else:
            print("\nNo valid credit spreads found matching the criteria.")

        print("\n" + "="*100)

    # Print trading statistics for multi-file analysis
    if results and len(csv_paths) > 1:
        print_trading_statistics(results, output_tz, len(csv_paths))

    # Strategy framework execution (when --strategy is used)
    if strategy and results:
        _execute_strategy_on_results(strategy, results, output_tz, logger, args, min_premium_diff)

    # Scale-in analysis (legacy path, only when --strategy is NOT used)
    if not strategy and args.scale_in_enabled and scale_in_config and results:
        print("\n" + "="*100)
        print("Running Scale-In on Breach Strategy Analysis...")
        print("="*100)

        results_by_date = defaultdict(lambda: {'put': [], 'call': []})

        for result in results:
            timestamp = result['timestamp']
            if hasattr(timestamp, 'date'):
                trading_date = timestamp.date()
            else:
                trading_date = pd.to_datetime(timestamp).date()

            opt_type = result.get('option_type', 'unknown').lower()
            if opt_type in ['put', 'call']:
                results_by_date[trading_date][opt_type].append(result)

        if args.db_path:
            scale_in_db_config = args.db_path
        else:
            scale_in_db_config = os.getenv('QUEST_DB_STRING', '') or os.getenv('QUESTDB_CONNECTION_STRING', '') or os.getenv('QUESTDB_URL', '')
        db = StockQuestDB(
            scale_in_db_config,
            enable_cache=not args.no_cache,
            logger=logger
        )

        try:
            scale_in_results = []

            for trading_date, type_results in sorted(results_by_date.items()):
                for opt_type in ['put', 'call']:
                    day_results = type_results[opt_type]
                    if not day_results:
                        continue

                    first_result = day_results[0]
                    prev_close = first_result.get('prev_close')
                    current_close = first_result.get('current_close')

                    if prev_close is None or current_close is None:
                        logger.debug(f"Skipping {trading_date} {opt_type}: missing price data")
                        continue

                    trading_datetime = datetime.combine(trading_date, datetime.min.time())
                    if output_tz:
                        try:
                            trading_datetime = output_tz.localize(trading_datetime)
                        except AttributeError:
                            trading_datetime = trading_datetime.replace(tzinfo=output_tz)

                    try:
                        scale_in_result = await analyze_scale_in_trade(
                            db=db,
                            trading_date=trading_datetime,
                            option_type=opt_type,
                            prev_close=prev_close,
                            current_close=current_close,
                            scale_in_config=scale_in_config,
                            logger=logger,
                            intraday_prices=None
                        )
                        scale_in_results.append(scale_in_result)
                    except Exception as e:
                        logger.warning(f"Error analyzing scale-in for {trading_date} {opt_type}: {e}")
                        continue

            if scale_in_results:
                aggregate_stats = aggregate_scale_in_results(scale_in_results, output_tz)
                print_scale_in_statistics(
                    aggregate_stats=aggregate_stats,
                    scale_in_results=scale_in_results,
                    scale_in_config=scale_in_config,
                    comparison_results=results,
                    summary_only=getattr(args, 'scale_in_summary_only', False)
                )
            else:
                print("No scale-in trades could be analyzed (missing price data)")

        finally:
            await db.close()

    # Tiered investment analysis (legacy path, only when --strategy is NOT used)
    if not strategy and args.tiered_enabled and tiered_config and results:
        print("\n" + "="*100)
        print("Running Tiered Investment Strategy Analysis...")
        print("="*100)

        tiered_results_by_date = defaultdict(lambda: {'put': [], 'call': []})

        for result in results:
            timestamp = result['timestamp']
            if hasattr(timestamp, 'date'):
                trading_date = timestamp.date()
            else:
                trading_date = pd.to_datetime(timestamp).date()

            opt_type = result.get('option_type', 'unknown').lower()
            if opt_type in ['put', 'call']:
                tiered_results_by_date[trading_date][opt_type].append(result)

        tiered_results = []

        for trading_date, type_results in sorted(tiered_results_by_date.items()):
            for opt_type in ['put', 'call']:
                day_results = type_results[opt_type]
                if not day_results:
                    continue

                first_result = day_results[0]
                prev_close = first_result.get('prev_close')
                current_close = first_result.get('current_close')

                if prev_close is None or current_close is None:
                    logger.debug(f"Skipping tiered {trading_date} {opt_type}: missing price data")
                    continue

                trading_datetime = datetime.combine(trading_date, datetime.min.time())
                if output_tz:
                    try:
                        trading_datetime = output_tz.localize(trading_datetime)
                    except AttributeError:
                        trading_datetime = trading_datetime.replace(tzinfo=output_tz)

                try:
                    trade_state = initialize_tiered_trade(
                        trading_date=trading_datetime,
                        option_type=opt_type,
                        prev_close=prev_close,
                        config=tiered_config,
                        day_results=day_results,
                        min_premium_diff=min_premium_diff,
                        max_credit_width_ratio=args.max_credit_width_ratio,
                        min_contract_price=args.min_contract_price,
                        max_strike_distance_pct=args.max_strike_distance_pct,
                        logger=logger,
                    )

                    trade_state = calculate_all_tiers_pnl(trade_state, current_close)

                    single_entry_pnl = None
                    for r in day_results:
                        pnl = r.get('actual_pnl_per_share')
                        bs = r.get('best_spread', {})
                        n = bs.get('num_contracts') or 0
                        if pnl is not None and n > 0:
                            single_entry_pnl = pnl * n * 100
                            break

                    summary = generate_tiered_summary(trade_state, single_entry_pnl)
                    tiered_results.append({
                        'trade_state': trade_state,
                        'summary': summary,
                    })
                except Exception as e:
                    logger.warning(f"Error analyzing tiered for {trading_date} {opt_type}: {e}")
                    continue

        if tiered_results:
            tiered_aggregate = aggregate_tiered_results(tiered_results, output_tz)
            print_tiered_statistics(
                aggregate_stats=tiered_aggregate,
                tiered_results=tiered_results,
                config=tiered_config,
                comparison_results=results,
                summary_only=getattr(args, 'tiered_summary_only', False),
            )
        else:
            print("No tiered trades could be analyzed (missing price data or no qualifying tiers)")

    # Generate histogram if requested
    if args.histogram and results and len(csv_paths) > 1:
        print("\nGenerating hourly analysis histogram...")
        generate_hourly_histogram(results, args.histogram_output, output_tz)
    elif args.histogram and len(csv_paths) == 1:
        print("\nNote: Histogram generation is most useful with multiple input files.")
        if results:
            generate_hourly_histogram(results, args.histogram_output, output_tz)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
