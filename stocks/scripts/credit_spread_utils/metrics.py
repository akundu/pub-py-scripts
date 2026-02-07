"""
Metrics computation and statistics reporting for credit spread analysis.

Functions for computing trading metrics, filtering results, and
printing comprehensive statistics summaries.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict
import pandas as pd
from .timezone_utils import format_timestamp

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def compute_metrics(results: List[Dict]) -> dict:
    """Compute aggregate trading metrics from a list of interval results."""
    if not results:
        return {
            'total_trades': 0, 'win_rate': 0.0, 'total_credits': 0.0,
            'total_gains': 0.0, 'total_losses': 0.0, 'net_pnl': 0.0,
            'roi': 0.0, 'profit_factor': 0.0,
        }

    successful_trades = []
    failed_trades = []
    pending_trades = []

    for result in results:
        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or result['best_spread'].get('net_credit_per_contract', 0)
        max_loss = result['best_spread'].get('total_max_loss') or result['best_spread'].get('max_loss_per_contract', 0)

        actual_pnl_per_share = result.get('actual_pnl_per_share')
        num_contracts = result['best_spread'].get('num_contracts', 1)
        if actual_pnl_per_share is not None and num_contracts:
            actual_pnl = actual_pnl_per_share * num_contracts * 100
        else:
            actual_pnl = None

        trade_info = {'credit': credit, 'max_loss': max_loss, 'actual_pnl': actual_pnl}

        if backtest_result is True:
            successful_trades.append(trade_info)
        elif backtest_result is False:
            failed_trades.append(trade_info)
        else:
            pending_trades.append(trade_info)

    total_trades = len(results)
    num_successful = len(successful_trades)
    num_failed = len(failed_trades)
    testable_trades = num_successful + num_failed
    win_rate = (num_successful / testable_trades * 100) if testable_trades > 0 else 0

    total_credits = sum(t['credit'] for t in successful_trades + failed_trades + pending_trades)

    total_gains = 0.0
    for t in successful_trades:
        if t['actual_pnl'] is not None:
            total_gains += t['actual_pnl']
        else:
            total_gains += t['credit']

    total_losses = 0.0
    for t in failed_trades:
        if t['actual_pnl'] is not None:
            total_losses += abs(t['actual_pnl'])
        else:
            total_losses += t['max_loss']

    net_pnl = total_gains - total_losses

    total_risk_deployed = sum(t['max_loss'] for t in successful_trades + failed_trades + pending_trades)
    roi = (net_pnl / total_risk_deployed * 100) if total_risk_deployed > 0 else 0

    if total_losses > 0:
        profit_factor = total_gains / total_losses
    elif total_gains > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0.0

    return {
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'total_credits': round(total_credits, 2),
        'total_gains': round(total_gains, 2),
        'total_losses': round(total_losses, 2),
        'net_pnl': round(net_pnl, 2),
        'roi': round(roi, 2),
        'profit_factor': round(profit_factor, 4),
    }


def filter_top_n_per_day(results: List[Dict], top_n: int) -> List[Dict]:
    """Filter results to keep only top N spreads per day (by max credit).

    Args:
        results: List of result dictionaries
        top_n: Number of top spreads to keep per day

    Returns:
        Filtered list containing only top N spreads per day
    """
    if not results or top_n is None:
        return results

    # Group results by day
    from collections import defaultdict
    by_day = defaultdict(list)

    for result in results:
        timestamp = result['timestamp']
        # Get date (without time)
        if hasattr(timestamp, 'date'):
            day = timestamp.date()
        else:
            day = pd.to_datetime(timestamp).date()
        by_day[day].append(result)

    # For each day, keep only top N by max credit
    filtered_results = []
    for day, day_results in sorted(by_day.items()):
        # Sort by max credit (descending)
        sorted_day_results = sorted(
            day_results,
            key=lambda x: x['best_spread'].get('total_credit') or x['best_spread'].get('net_credit_per_contract', 0),
            reverse=True
        )
        # Keep top N
        filtered_results.extend(sorted_day_results[:top_n])

    return filtered_results


def print_trading_statistics(results: List[Dict], output_tz, total_files_processed: int = 0):
    """Print comprehensive trading statistics for multi-file analysis."""
    if not results:
        print("No results to analyze.")
        return

    # Collect statistics
    total_trades = len(results)
    unique_dates = set()
    unique_source_files = set()

    successful_trades = []
    failed_trades = []
    pending_trades = []

    for result in results:
        timestamp = result['timestamp']
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)
        unique_dates.add(timestamp.date())

        # Track source file if available
        source_file = result.get('source_file')
        if source_file:
            unique_source_files.add(source_file)

        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or (result['best_spread'].get('net_credit_per_contract', 0))
        max_loss = result['best_spread'].get('total_max_loss') or (result['best_spread'].get('max_loss_per_contract', 0))

        # Get actual P&L if available (from force close calculation)
        actual_pnl_per_share = result.get('actual_pnl_per_share')
        num_contracts = result['best_spread'].get('num_contracts', 1)
        if actual_pnl_per_share is not None and num_contracts:
            actual_pnl = actual_pnl_per_share * num_contracts * 100  # per contract = per share * 100
        else:
            actual_pnl = None

        trade_info = {
            'timestamp': timestamp,
            'credit': credit,
            'max_loss': max_loss,
            'actual_pnl': actual_pnl,
            'option_type': result.get('option_type', 'UNKNOWN'),
            'spread': f"${result['best_spread']['short_strike']:.2f}/${result['best_spread']['long_strike']:.2f}",
            'contracts': result['best_spread'].get('num_contracts', 1)
        }

        if backtest_result is True:
            successful_trades.append(trade_info)
        elif backtest_result is False:
            failed_trades.append(trade_info)
        else:
            pending_trades.append(trade_info)

    # Calculate statistics
    num_unique_days = len(unique_dates)
    num_successful = len(successful_trades)
    num_failed = len(failed_trades)
    num_pending = len(pending_trades)

    # Calculate financial metrics
    # Use actual P&L if available (from force close), otherwise use credit/max_loss
    total_credits = sum(t['credit'] for t in successful_trades + failed_trades + pending_trades)

    # Calculate actual gains and losses
    total_gains = 0
    total_losses = 0

    for t in successful_trades:
        if t['actual_pnl'] is not None:
            total_gains += t['actual_pnl']
        else:
            total_gains += t['credit']

    for t in failed_trades:
        if t['actual_pnl'] is not None:
            # actual_pnl is negative for losses
            total_losses += abs(t['actual_pnl'])
        else:
            total_losses += t['max_loss']

    net_pnl = total_gains - total_losses

    # Win rate (excluding pending)
    testable_trades = num_successful + num_failed
    win_rate = (num_successful / testable_trades * 100) if testable_trades > 0 else 0

    # Averages
    avg_credit_per_trade = total_credits / total_trades if total_trades > 0 else 0
    avg_gain_per_win = sum(t['credit'] for t in successful_trades) / num_successful if num_successful > 0 else 0
    avg_loss_per_loss = sum(t['max_loss'] for t in failed_trades) / num_failed if num_failed > 0 else 0

    # Best/worst trades
    all_completed = successful_trades + failed_trades
    if all_completed:
        best_trade = max(all_completed, key=lambda x: x['credit'] if x in successful_trades else -x['max_loss'])
        worst_trade = min(all_completed, key=lambda x: x['credit'] if x in successful_trades else -x['max_loss'])

    # Total risk deployed (sum of max losses for all trades)
    total_risk_deployed = sum(t['max_loss'] for t in successful_trades + failed_trades + pending_trades)

    # ROI
    roi = (net_pnl / total_risk_deployed * 100) if total_risk_deployed > 0 else 0

    # Print statistics
    print("\n" + "="*100)
    print("MULTI-FILE TRADING STATISTICS")
    print("="*100)

    # File processing statistics
    num_files_with_results = len(unique_source_files)
    if total_files_processed > 0:
        print(f"\n\U0001f4c1 FILE PROCESSING:")
        print(f"  Total Files Processed: {total_files_processed}")
        print(f"  Files with Valid Results: {num_files_with_results} ({num_files_with_results/total_files_processed*100:.1f}%)")
        if total_files_processed > num_files_with_results:
            files_no_results = total_files_processed - num_files_with_results
            print(f"  Files with No Valid Spreads: {files_no_results} ({files_no_results/total_files_processed*100:.1f}%)")
            print(f"    (likely filtered out by: credit-width ratio, strike distance, or min price)")

    print(f"\n\U0001f4ca TRADING ACTIVITY:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Unique Trading Days: {num_unique_days}")
    print(f"  Average Trades per Day: {total_trades/num_unique_days:.1f}")

    print(f"\n\u2705 TRADE OUTCOMES:")
    print(f"  Successful: {num_successful} ({num_successful/total_trades*100:.1f}%)")
    print(f"  Failed: {num_failed} ({num_failed/total_trades*100:.1f}%)")
    print(f"  Pending: {num_pending} ({num_pending/total_trades*100:.1f}%)")
    if testable_trades > 0:
        print(f"  Win Rate (excl. pending): {win_rate:.1f}% ({num_successful}/{testable_trades})")

    print(f"\n\U0001f4b0 FINANCIAL PERFORMANCE:")
    print(f"  Total Credits Collected: ${total_credits:,.2f}")
    print(f"  Total Gains (wins only): ${total_gains:,.2f}")
    print(f"  Total Losses (failures): ${total_losses:,.2f}")
    print(f"  Net P&L: ${net_pnl:,.2f}", end="")
    if net_pnl >= 0:
        print(" \u2713")
    else:
        print(" \u2717")

    # Calculate PUT vs CALL breakdown
    put_trades = [t for t in successful_trades + failed_trades + pending_trades if t['option_type'].upper() == 'PUT']
    call_trades = [t for t in successful_trades + failed_trades + pending_trades if t['option_type'].upper() == 'CALL']

    put_credits = sum(t['credit'] for t in put_trades)
    call_credits = sum(t['credit'] for t in call_trades)

    # Calculate PUT P&L
    put_gains = 0
    put_losses = 0
    for t in put_trades:
        if t in successful_trades:
            if t['actual_pnl'] is not None:
                put_gains += t['actual_pnl']
            else:
                put_gains += t['credit']
        elif t in failed_trades:
            if t['actual_pnl'] is not None:
                put_losses += abs(t['actual_pnl'])
            else:
                put_losses += t['max_loss']
    put_net_pnl = put_gains - put_losses

    # Calculate CALL P&L
    call_gains = 0
    call_losses = 0
    for t in call_trades:
        if t in successful_trades:
            if t['actual_pnl'] is not None:
                call_gains += t['actual_pnl']
            else:
                call_gains += t['credit']
        elif t in failed_trades:
            if t['actual_pnl'] is not None:
                call_losses += abs(t['actual_pnl'])
            else:
                call_losses += t['max_loss']
    call_net_pnl = call_gains - call_losses

    # PUT vs CALL breakdown
    print(f"\n\U0001f4ca PUT vs CALL BREAKDOWN:")
    print(f"  {'Metric':<25} {'PUT':<20} {'CALL':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'Trades':<25} {len(put_trades):<20} {len(call_trades):<20}")
    print(f"  {'Total Credits':<25} ${put_credits:>18,.2f} ${call_credits:>18,.2f}")
    print(f"  {'Total Gains':<25} ${put_gains:>18,.2f} ${call_gains:>18,.2f}")
    print(f"  {'Total Losses':<25} ${put_losses:>18,.2f} ${call_losses:>18,.2f}")
    print(f"  {'Net P&L':<25} ${put_net_pnl:>18,.2f} ${call_net_pnl:>18,.2f}")

    # Calculate win rates for PUT and CALL
    put_successful = [t for t in put_trades if t in successful_trades]
    put_failed = [t for t in put_trades if t in failed_trades]
    put_testable = len(put_successful) + len(put_failed)
    put_win_rate = (len(put_successful) / put_testable * 100) if put_testable > 0 else 0

    call_successful = [t for t in call_trades if t in successful_trades]
    call_failed = [t for t in call_trades if t in failed_trades]
    call_testable = len(call_successful) + len(call_failed)
    call_win_rate = (len(call_successful) / call_testable * 100) if call_testable > 0 else 0

    print(f"  {'Win Rate':<25} {put_win_rate:>18.1f}% {call_win_rate:>18.1f}%")

    print(f"\n\U0001f4c8 AVERAGES:")
    print(f"  Average Credit per Trade: ${avg_credit_per_trade:,.2f}")
    if num_successful > 0:
        print(f"  Average Gain per Win: ${avg_gain_per_win:,.2f}")
    if num_failed > 0:
        print(f"  Average Loss per Loss: ${avg_loss_per_loss:,.2f}")

    print(f"\n\U0001f3af RISK METRICS:")
    print(f"  Total Risk Deployed: ${total_risk_deployed:,.2f}")
    print(f"  Return on Risk (ROI): {roi:+.2f}%")
    if testable_trades > 0:
        expectancy = (avg_gain_per_win * win_rate/100) - (avg_loss_per_loss * (100-win_rate)/100)
        print(f"  Expectancy per Trade: ${expectancy:,.2f}")

    if all_completed:
        print(f"\n\U0001f3c6 BEST/WORST TRADES:")
        if best_trade in successful_trades:
            print(f"  Best Trade: ${best_trade['credit']:,.2f} credit on {best_trade['timestamp'].strftime('%Y-%m-%d %H:%M')} ({best_trade['option_type']} {best_trade['spread']})")
        if worst_trade in failed_trades:
            print(f"  Worst Trade: -${worst_trade['max_loss']:,.2f} loss on {worst_trade['timestamp'].strftime('%Y-%m-%d %H:%M')} ({worst_trade['option_type']} {worst_trade['spread']})")

    # Analyze hourly performance and 10-minute blocks
    if results:
        print_hourly_summary(results, output_tz)
        print_10min_block_breakdown(results, output_tz)

    print("\n" + "="*100)


def print_hourly_summary(results: List[Dict], output_tz):
    """Print hourly summary table showing all hours with their performance stats."""
    if not results:
        return

    # Group by hour
    hourly_data = defaultdict(lambda: {'success': 0, 'failure': 0, 'pending': 0, 'total': 0, 'total_credit': 0, 'total_loss': 0})

    for result in results:
        timestamp = result['timestamp']
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)

        hour = timestamp.hour

        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or (result['best_spread'].get('net_credit_per_contract', 0))
        max_loss = result['best_spread'].get('total_max_loss') or (result['best_spread'].get('max_loss_per_contract', 0))

        hourly_data[hour]['total'] += 1
        hourly_data[hour]['total_credit'] += credit
        if backtest_result is True:
            hourly_data[hour]['success'] += 1
        elif backtest_result is False:
            hourly_data[hour]['failure'] += 1
            hourly_data[hour]['total_loss'] += max_loss
        else:
            hourly_data[hour]['pending'] += 1

    # Sort hours and print summary table
    sorted_hours = sorted(hourly_data.keys())

    if not sorted_hours:
        return

    print(f"\n\u23f0 HOURLY PERFORMANCE SUMMARY:")
    print("-" * 100)
    print(f"{'Hour':<8} {'Trades':<10} {'Success':<12} {'Failure':<12} {'Pending':<12} {'Win Rate':<12} {'Net P&L':<15}")
    print("-" * 100)

    for hour in sorted_hours:
        data = hourly_data[hour]
        testable = data['success'] + data['failure']
        win_rate = (data['success'] / testable * 100) if testable > 0 else 0
        net_pnl = data['total_credit'] - data['total_loss']

        print(f"{hour:02d}:00    {data['total']:<10} {data['success']:<12} {data['failure']:<12} {data['pending']:<12} {win_rate:>6.1f}%{'':<5} ${net_pnl:>12,.2f}")

    print("-" * 100)


def print_10min_block_breakdown(results: List[Dict], output_tz):
    """Print 10-minute block breakdown for the best performing hours."""
    if not results:
        return

    # Group by hour and 10-minute block
    block_data = defaultdict(lambda: {'success': 0, 'failure': 0, 'pending': 0, 'total': 0, 'total_credit': 0, 'total_loss': 0})

    for result in results:
        timestamp = result['timestamp']
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)

        hour = timestamp.hour
        minute = timestamp.minute
        # Round down to nearest 10-minute block (0, 10, 20, 30, 40, 50)
        block_minute = (minute // 10) * 10
        block_key = (hour, block_minute)

        backtest_result = result.get('backtest_successful')
        credit = result['best_spread'].get('total_credit') or (result['best_spread'].get('net_credit_per_contract', 0))
        max_loss = result['best_spread'].get('total_max_loss') or (result['best_spread'].get('max_loss_per_contract', 0))

        block_data[block_key]['total'] += 1
        block_data[block_key]['total_credit'] += credit
        if backtest_result is True:
            block_data[block_key]['success'] += 1
        elif backtest_result is False:
            block_data[block_key]['failure'] += 1
            block_data[block_key]['total_loss'] += max_loss
        else:
            block_data[block_key]['pending'] += 1

    # Calculate hourly aggregates to find best hours
    hourly_aggregates = defaultdict(lambda: {'total': 0, 'success': 0, 'failure': 0, 'total_credit': 0, 'total_loss': 0})

    for (hour, block_min), data in block_data.items():
        hourly_aggregates[hour]['total'] += data['total']
        hourly_aggregates[hour]['success'] += data['success']
        hourly_aggregates[hour]['failure'] += data['failure']
        hourly_aggregates[hour]['total_credit'] += data['total_credit']
        hourly_aggregates[hour]['total_loss'] += data['total_loss']

    # Find top 3 hours by total trades (or by success rate if tied)
    top_hours = sorted(
        hourly_aggregates.items(),
        key=lambda x: (x[1]['total'], x[1]['success'] / max(x[1]['success'] + x[1]['failure'], 1)),
        reverse=True
    )[:3]

    if not top_hours:
        return

    print(f"\n\u23f0 10-MINUTE BLOCK BREAKDOWN (Top {len(top_hours)} Hours):")
    print("-" * 100)

    for hour, hour_stats in top_hours:
        testable = hour_stats['success'] + hour_stats['failure']
        win_rate = (hour_stats['success'] / testable * 100) if testable > 0 else 0
        net_pnl = hour_stats['total_credit'] - hour_stats['total_loss']

        print(f"\n  Hour {hour:02d}:00 - {hour_stats['total']} trades | {hour_stats['success']}\u2713 / {hour_stats['failure']}\u2717 ({win_rate:.1f}% win rate) | Net P&L: ${net_pnl:+,.2f}")

        # Get all 10-minute blocks for this hour
        hour_blocks = [(h, bm, data) for (h, bm), data in block_data.items() if h == hour]
        hour_blocks.sort(key=lambda x: x[1])  # Sort by block minute

        if hour_blocks:
            print(f"    {'Block':<12} {'Trades':<8} {'Success':<10} {'Failure':<10} {'Win Rate':<12} {'Net P&L':<15}")
            print(f"    {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")

            for h, block_min, data in hour_blocks:
                block_testable = data['success'] + data['failure']
                block_win_rate = (data['success'] / block_testable * 100) if block_testable > 0 else 0
                block_net_pnl = data['total_credit'] - data['total_loss']
                block_label = f"{h:02d}:{block_min:02d}-{block_min+9:02d}"

                print(f"    {block_label:<12} {data['total']:<8} {data['success']:<10} {data['failure']:<10} {block_win_rate:>6.1f}%{'':<5} ${block_net_pnl:>12,.2f}")


def generate_hourly_histogram(results: List[Dict], output_path: str, output_tz):
    """Generate histogram showing hourly performance of credit spreads."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Cannot generate histogram.")
        return

    if not results:
        print("No results to generate histogram.")
        return

    # Group results by hour
    hourly_data = defaultdict(lambda: {'success': 0, 'failure': 0, 'pending': 0, 'total': 0})

    for result in results:
        timestamp = result['timestamp']
        # Convert to output timezone for display
        if hasattr(timestamp, 'astimezone'):
            timestamp = timestamp.astimezone(output_tz)
        hour = timestamp.hour

        backtest_result = result.get('backtest_successful')
        hourly_data[hour]['total'] += 1

        if backtest_result is True:
            hourly_data[hour]['success'] += 1
        elif backtest_result is False:
            hourly_data[hour]['failure'] += 1
        else:
            hourly_data[hour]['pending'] += 1

    # Prepare data for plotting
    hours = sorted(hourly_data.keys())
    successes = [hourly_data[h]['success'] for h in hours]
    failures = [hourly_data[h]['failure'] for h in hours]
    totals = [hourly_data[h]['total'] for h in hours]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Stacked bar chart of successes/failures
    x = range(len(hours))
    ax1.bar(x, successes, label='Success \u2713', color='green', alpha=0.7)
    ax1.bar(x, failures, bottom=successes, label='Failure \u2717', color='red', alpha=0.7)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Count')
    ax1.set_title('Credit Spread Performance by Hour')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success rate percentage
    success_rates = []
    for h in hours:
        testable = hourly_data[h]['success'] + hourly_data[h]['failure']
        if testable > 0:
            success_rates.append((hourly_data[h]['success'] / testable) * 100)
        else:
            success_rates.append(0)

    ax2.bar(x, success_rates, color='blue', alpha=0.7)
    ax2.axhline(y=50, color='gray', linestyle='--', label='50% baseline')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Success Rate by Hour')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add annotations showing counts
    for i, (h, s, f) in enumerate(zip(hours, successes, failures)):
        ax1.text(i, s + f + 0.5, f"{s+f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_path}")

    # Print hourly summary table
    print("\n" + "="*80)
    print("HOURLY PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Hour':<8} {'Total':<8} {'Success':<10} {'Failure':<10} {'Success Rate':<15}")
    print("-"*80)

    for h in hours:
        total = hourly_data[h]['total']
        success = hourly_data[h]['success']
        failure = hourly_data[h]['failure']
        testable = success + failure
        success_rate = (success / testable * 100) if testable > 0 else 0

        print(f"{h:02d}:00    {total:<8} {success:<10} {failure:<10} {success_rate:>6.1f}%")

    print("="*80)
