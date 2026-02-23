"""
Risk gradient analysis utilities.

Two-part analysis:
1. Query historical max price movements from QuestDB daily_prices table
2. Generate risk gradient values from the "zero risk" safe point and test
   shifting toward the money to see the risk/reward tradeoff

Extracted from scripts/ndx_risk_gradient_analysis.py for reuse via
analyze_credit_spread_intervals.py --mode risk-gradient.

Depends on: common.questdb_db.StockQuestDB, common.logging_utils.get_logger
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# HISTORICAL MOVEMENT QUERIES
# ============================================================================

async def query_historical_max_movements(
    db,
    ticker: str,
    lookback_days: int,
    logger
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Query daily_prices table for historical maximum price movements.

    Returns two metrics:
    1. Intraday Extreme: (prev_close - low) / prev_close for PUTs,
       (high - prev_close) / prev_close for CALLs
    2. Close-to-Close: (prev_close - close) / prev_close for down days,
       (close - prev_close) / prev_close for up days

    Args:
        db: Database connection (StockQuestDB instance)
        ticker: Ticker symbol (e.g., 'NDX' — no I: prefix)
        lookback_days: Number of days to look back
        logger: Logger instance

    Returns:
        Dict with 'intraday' and 'close_to_close' metrics, or None on failure.
    """
    import numpy as np
    logger.info(f"Querying {lookback_days}-day historical max movements for {ticker}")

    query = """
    SELECT
        date,
        open,
        high,
        low,
        close
    FROM daily_prices
    WHERE ticker = $1
    ORDER BY date DESC
    LIMIT $2
    """

    try:
        result = await db.execute_select_sql(query, (ticker, lookback_days + 5))

        if result is None or result.empty:
            logger.warning(f"No data returned for {ticker} with {lookback_days}-day lookback")
            return None

        df = result.sort_values('date').reset_index(drop=True)

        if len(df) < 2:
            logger.warning(f"Not enough data for {ticker} (only {len(df)} rows)")
            return None

        df['prev_close'] = df['close'].shift(1)
        df = df.dropna(subset=['prev_close'])

        if len(df) > lookback_days:
            df = df.tail(lookback_days)

        if len(df) == 0:
            logger.warning(f"No valid data after filtering for {ticker}")
            return None

        # Calculate metrics
        df['intraday_down'] = (df['prev_close'] - df['low']) / df['prev_close'] * 100
        df['intraday_up'] = (df['high'] - df['prev_close']) / df['prev_close'] * 100
        df['close_change'] = (df['close'] - df['prev_close']) / df['prev_close'] * 100
        df['close_down'] = df['close_change'].apply(lambda x: abs(x) if x < 0 else 0)
        df['close_up'] = df['close_change'].apply(lambda x: x if x > 0 else 0)

        metrics = {
            'intraday': {
                'put': float(df['intraday_down'].max()),
                'call': float(df['intraday_up'].max()),
            },
            'close_to_close': {
                'put': float(df['close_down'].max()),
                'call': float(df['close_up'].max()),
            },
            'stats': {
                'avg_intraday_down': float(df['intraday_down'].mean()),
                'avg_intraday_up': float(df['intraday_up'].mean()),
                'p95_intraday_down': float(np.percentile(df['intraday_down'], 95)),
                'p95_intraday_up': float(np.percentile(df['intraday_up'], 95)),
                'trading_days': len(df),
            }
        }

        logger.info(f"  Intraday Extreme - PUT: {metrics['intraday']['put']:.2f}%, CALL: {metrics['intraday']['call']:.2f}%")
        logger.info(f"  Close-to-Close - PUT: {metrics['close_to_close']['put']:.2f}%, CALL: {metrics['close_to_close']['call']:.2f}%")
        logger.info(f"  Trading days analyzed: {metrics['stats']['trading_days']}")

        return metrics

    except Exception as e:
        logger.error(f"Error querying historical movements: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        raise


# ============================================================================
# RISK GRADIENT GENERATION
# ============================================================================

def generate_risk_gradient_values(
    safe_point_put: float,
    safe_point_call: float,
    gradient_steps: int,
    step_size: float
) -> List[Tuple[str, float, float, str]]:
    """
    Generate risk gradient values starting from the safe point and moving toward the money.

    Args:
        safe_point_put: PUT safe point as percentage (e.g., 2.51 for 2.51%)
        safe_point_call: CALL safe point as percentage
        gradient_steps: Number of steps from safe point
        step_size: Step size in decimal (0.0025 = 0.25%)

    Returns:
        List of tuples: (percent_beyond_string, put_pct, call_pct, risk_label)
    """
    risk_labels = [
        "Zero historical risk",
        "Minimal risk",
        "Low risk",
        "Moderate risk",
        "Higher risk",
        "High risk",
        "Very high risk",
        "Extreme risk",
        "Maximum risk",
    ]

    values = []
    for i in range(gradient_steps):
        offset = i * step_size
        put_pct = safe_point_put / 100 - offset
        call_pct = safe_point_call / 100 - offset

        if put_pct <= 0 or call_pct <= 0:
            break

        percent_beyond = f"{put_pct:.4f}:{call_pct:.4f}"
        risk_label = risk_labels[min(i, len(risk_labels) - 1)]

        values.append((percent_beyond, put_pct * 100, call_pct * 100, risk_label))

    return values


def create_grid_config(
    gradient_values: List[Tuple[str, float, float, str]],
    args,
    lookback_days: int,
    metric_type: str,
    safe_point_put: float,
    safe_point_call: float
) -> Dict:
    """
    Create grid config JSON for the risk gradient analysis.

    Args:
        gradient_values: List of (percent_beyond, put_pct, call_pct, risk_label)
        args: Command line arguments
        lookback_days: Lookback period in days
        metric_type: 'intraday' or 'close_to_close'
        safe_point_put: PUT safe point percentage
        safe_point_call: CALL safe point percentage

    Returns:
        Grid config dictionary
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    csv_dir = getattr(args, 'csv_dir', '../options_csv_output')
    risk_cap = getattr(args, 'risk_cap', 500000)
    min_trading_hour = getattr(args, 'min_trading_hour', 9)
    max_trading_hour = getattr(args, 'max_trading_hour', 12)
    step_size = getattr(args, 'step_size', 0.0025)

    # Resolve ticker — strip I: prefix for options
    ticker = getattr(args, 'underlying_ticker', None) or getattr(args, 'ticker', 'NDX')
    ticker_clean = ticker.replace('I:', '')

    config = {
        "_comment": f"NDX Risk Gradient Analysis - {lookback_days}-day {metric_type} metric",
        "_safe_point": {
            "put_pct": safe_point_put,
            "call_pct": safe_point_call,
            "lookback_days": lookback_days,
            "metric_type": metric_type
        },
        "_gradient_legend": [
            {"offset": f"-{i * step_size * 100:.2f}%", "risk": gv[3]}
            for i, gv in enumerate(gradient_values)
        ],
        "grid_params": {
            "percent_beyond": [gv[0] for gv in gradient_values]
        },
        "fixed_params": {
            "csv_dir": csv_dir,
            "underlying_ticker": ticker_clean,
            "option_type": "both",
            "min_spread_width": 10,
            "max_spread_width": "30:30",
            "risk_cap": risk_cap,
            "profit_target_pct": 0.90,
            "min_trading_hour": min_trading_hour,
            "max_trading_hour": max_trading_hour,
            "min_premium_diff": "0.50:0.50",
            "output_timezone": "America/Los_Angeles",
            "db_path": "$QUEST_DB_STRING",
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d')
        }
    }

    return config


# ============================================================================
# DISPLAY / PRINTING FUNCTIONS
# ============================================================================

def print_safe_points_comparison(
    all_metrics: Dict[int, Dict[str, Dict[str, float]]],
    logger
):
    """Print a comparison table of safe points across different lookback periods."""

    print("\n" + "=" * 80)
    print("SAFE POINTS COMPARISON")
    print("=" * 80)

    lookback_days = sorted(all_metrics.keys())

    header = f"{'Metric':<25}"
    for days in lookback_days:
        header += f"  {days}-Day PUT  {days}-Day CALL"
    print(header)
    print("-" * 80)

    row = f"{'Intraday Extreme':<25}"
    for days in lookback_days:
        metrics = all_metrics[days]
        row += f"  {metrics['intraday']['put']:>8.2f}%  {metrics['intraday']['call']:>10.2f}%"
    print(row)

    row = f"{'Close-to-Close':<25}"
    for days in lookback_days:
        metrics = all_metrics[days]
        row += f"  {metrics['close_to_close']['put']:>8.2f}%  {metrics['close_to_close']['call']:>10.2f}%"
    print(row)

    print()
    print("Additional Statistics:")
    for days in lookback_days:
        stats = all_metrics[days]['stats']
        print(f"  {days}-day period ({stats['trading_days']} trading days):")
        print(f"    Avg intraday down: {stats['avg_intraday_down']:.2f}%")
        print(f"    Avg intraday up: {stats['avg_intraday_up']:.2f}%")
        print(f"    95th percentile down: {stats['p95_intraday_down']:.2f}%")
        print(f"    95th percentile up: {stats['p95_intraday_up']:.2f}%")

    print("=" * 80)


def print_gradient_preview(
    gradient_values: List[Tuple[str, float, float, str]],
    lookback_days: int,
    metric_type: str,
    safe_point_put: float,
    safe_point_call: float
):
    """Print a preview of the risk gradient that will be tested."""

    print(f"\nRisk Gradient Preview ({lookback_days}-day {metric_type}):")
    print(f"Safe Point: PUT {safe_point_put:.2f}% / CALL {safe_point_call:.2f}%")
    print()
    print(f"{'Offset':<15} {'PUT %':<10} {'CALL %':<10} {'Risk Level':<25}")
    print("-" * 60)

    for i, (percent_beyond, put_pct, call_pct, risk_label) in enumerate(gradient_values):
        offset = f"-{i * 0.25:.2f}%" if i > 0 else "Safe"
        print(f"{offset:<15} {put_pct:.2f}%{'':<5} {call_pct:.2f}%{'':<5} {risk_label:<25}")


# ============================================================================
# BACKTEST RUNNERS
# ============================================================================

async def run_backtest_for_config(
    config_path: str,
    output_path: str,
    processes: int,
    logger
) -> Optional[str]:
    """
    Run the credit spread analysis for a given config file.

    Args:
        config_path: Path to the grid config JSON
        output_path: Path for the output CSV
        processes: Number of parallel processes
        logger: Logger instance

    Returns:
        Path to output file if successful, None otherwise
    """
    import subprocess

    scripts_dir = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        str(scripts_dir / 'analyze_credit_spread_intervals.py'),
        '--grid-config', config_path,
        '--grid-output', output_path,
        '--grid-sort', 'win_rate',
        '--processes', str(processes),
        '--log-level', 'WARNING'
    ]

    logger.info(f"Running backtest: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(scripts_dir))

        if result.returncode != 0:
            logger.error(f"Backtest failed: {result.stderr}")
            return None

        logger.info(f"Backtest completed: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return None


async def run_detailed_backtest(
    config_path: str,
    ticker: str,
    processes: int,
    logger
) -> Optional[Dict]:
    """
    Run a detailed backtest and capture hourly/time-block statistics.

    Runs analyze_credit_spread_intervals.py and parses the output
    to extract hourly performance data.

    Args:
        config_path: Path to the config JSON
        ticker: Ticker symbol
        processes: Number of parallel processes
        logger: Logger instance

    Returns:
        Dictionary with hourly and time-block statistics, or None on failure
    """
    import subprocess
    import re

    scripts_dir = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        str(scripts_dir / 'analyze_credit_spread_intervals.py'),
        '--grid-config', config_path,
        '--grid-output', '/dev/null',
        '--processes', str(processes),
        '--log-level', 'WARNING'
    ]

    logger.info(f"Running detailed backtest: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(scripts_dir), timeout=600)

        output = result.stdout + result.stderr

        hourly_stats = {}
        time_block_stats = {}

        hourly_match = re.search(r'HOURLY PERFORMANCE SUMMARY:.*?(?=-{50,}|\n\n)', output, re.DOTALL)
        if hourly_match:
            hourly_section = hourly_match.group(0)
            for line in hourly_section.split('\n'):
                match = re.match(r'\s*(\d{2}):00\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+\$([\d,.-]+)', line)
                if match:
                    hour = int(match.group(1))
                    hourly_stats[hour] = {
                        'trades': int(match.group(2)),
                        'success': int(match.group(3)),
                        'failure': int(match.group(4)),
                        'pending': int(match.group(5)),
                        'win_rate': float(match.group(6)),
                        'net_pnl': float(match.group(7).replace(',', ''))
                    }

        return {
            'hourly': hourly_stats,
            'time_blocks': time_block_stats,
            'raw_output': output
        }

    except subprocess.TimeoutExpired:
        logger.error("Backtest timed out")
        return None
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return None


# ============================================================================
# RESULTS PARSING
# ============================================================================

def parse_and_display_results(
    results_path: str,
    gradient_values: List[Tuple[str, float, float, str]],
    lookback_days: int,
    metric_type: str,
    logger,
    step_size: float = 0.0025
):
    """
    Parse backtest results and display formatted risk gradient summary.

    Args:
        results_path: Path to the results CSV
        gradient_values: List of gradient values for labeling
        lookback_days: Lookback period
        metric_type: Metric type used
        logger: Logger instance
        step_size: Step size for offset calculation
    """
    import pandas as pd

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        logger.error(f"Error reading results: {e}")
        return

    print(f"\n{'=' * 110}")
    print(f"RISK GRADIENT RESULTS ({lookback_days}-day {metric_type})")
    print(f"{'=' * 110}")

    gradient_map = {gv[0]: (i, gv[1], gv[2], gv[3]) for i, gv in enumerate(gradient_values)}

    print(f"\n{'Offset':<10} {'PUT %':<8} {'CALL %':<8} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Win Rate':<10} {'Net P&L':<15} {'Risk Level':<20}")
    print("-" * 115)

    sorted_rows = []
    for _, row in df.iterrows():
        percent_beyond = row.get('percent_beyond', '')
        if percent_beyond in gradient_map:
            idx, put_pct, call_pct, risk_label = gradient_map[percent_beyond]
            sorted_rows.append((idx, row, put_pct, call_pct, risk_label))

    sorted_rows.sort(key=lambda x: x[0])

    for idx, row, put_pct, call_pct, risk_label in sorted_rows:
        offset = f"-{idx * step_size * 100:.2f}%" if idx > 0 else "Safe"

        trades = int(row.get('total_trades', 0))
        win_rate = float(row.get('win_rate', 0))
        net_pnl = float(row.get('net_pnl', 0))

        wins = int(round(trades * win_rate / 100))
        losses = trades - wins

        print(f"{offset:<10} {put_pct:.2f}%{'':<3} {call_pct:.2f}%{'':<3} {trades:<8} {wins:<8} {losses:<8} {win_rate:>6.1f}%{'':<3} ${net_pnl:>12,.2f}  {risk_label:<20}")

    print("-" * 115)

    if sorted_rows:
        print(f"\nSummary:")
        total_trades = sum(int(r[1].get('total_trades', 0)) for r in sorted_rows)
        total_net_pnl = sum(float(r[1].get('net_pnl', 0)) for r in sorted_rows)
        print(f"  Total trades across all gradient levels: {total_trades}")
        print(f"  Combined net P&L: ${total_net_pnl:,.2f}")


# ============================================================================
# TIME PERIOD ANALYSIS
# ============================================================================

def create_time_period_config(
    base_config: Dict,
    period_name: str,
    output_dir: Path
) -> Tuple[str, str, str]:
    """
    Create a config file for a specific time period based on a base config.

    Args:
        base_config: Base configuration dictionary
        period_name: Period name (3mo, 1mo, week1, week2, week3, week4)
        output_dir: Directory to save the config

    Returns:
        Tuple of (config_path, results_path, period_label)
    """
    end_date = datetime.now()

    period_map = {
        '3mo': (timedelta(days=90), '3-Month'),
        '1mo': (timedelta(days=30), '1-Month'),
        'week1': (timedelta(days=7), 'Week 1 (Most Recent)'),
        'week2': (timedelta(days=14), 'Week 2'),
        'week3': (timedelta(days=21), 'Week 3'),
        'week4': (timedelta(days=28), 'Week 4'),
    }

    if period_name not in period_map:
        raise ValueError(f"Unknown period: {period_name}")

    delta, label = period_map[period_name]

    if period_name.startswith('week'):
        week_num = int(period_name[-1])
        week_end = end_date - timedelta(days=(week_num - 1) * 7)
        week_start = week_end - timedelta(days=7)
    else:
        week_start = end_date - delta
        week_end = end_date

    config = json.loads(json.dumps(base_config))
    config['fixed_params']['start_date'] = week_start.strftime('%Y-%m-%d')
    config['fixed_params']['end_date'] = week_end.strftime('%Y-%m-%d')

    if 'grid_params' in config and 'percent_beyond' in config['grid_params']:
        percent_beyond = config['grid_params']['percent_beyond'][0]
        config['grid_params'] = {'percent_beyond': [percent_beyond]}

    config_filename = f"time_analysis_{period_name}.json"
    config_path = output_dir / config_filename
    results_path = output_dir / f"time_analysis_{period_name}_results.csv"

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return str(config_path), str(results_path), label


def print_time_of_day_summary(
    all_period_stats: Dict[str, Dict],
    logger
):
    """
    Print a formatted time-of-day performance summary across all periods.

    Args:
        all_period_stats: Dictionary mapping period names to their stats
        logger: Logger instance
    """
    if not all_period_stats:
        print("No time-of-day data available.")
        return

    print("\n" + "=" * 100)
    print("TIME-OF-DAY PERFORMANCE ANALYSIS")
    print("=" * 100)

    all_hours = set()
    for period, stats in all_period_stats.items():
        if 'hourly' in stats:
            all_hours.update(stats['hourly'].keys())

    if not all_hours:
        print("No hourly data available in the results.")
        return

    sorted_hours = sorted(all_hours)

    header = f"{'Hour':<8}"
    for period in all_period_stats.keys():
        header += f" {period:^25}"
    print(header)
    print("-" * (8 + len(all_period_stats) * 26))

    sub_header = f"{'':8}"
    for _ in all_period_stats.keys():
        sub_header += f" {'Trades':^8} {'WR':^6} {'Net P&L':^10}"
    print(sub_header)
    print("-" * (8 + len(all_period_stats) * 26))

    for hour in sorted_hours:
        row = f"{hour:02d}:00   "
        for period, stats in all_period_stats.items():
            hourly = stats.get('hourly', {})
            if hour in hourly:
                h = hourly[hour]
                row += f" {h['trades']:^8} {h['win_rate']:>5.1f}% ${h['net_pnl']:>9,.0f}"
            else:
                row += f" {'-':^8} {'-':^6} {'-':^10}"
        print(row)

    print("-" * (8 + len(all_period_stats) * 26))

    totals_row = f"{'TOTAL':8}"
    for period, stats in all_period_stats.items():
        hourly = stats.get('hourly', {})
        total_trades = sum(h['trades'] for h in hourly.values())
        total_wins = sum(h['success'] for h in hourly.values())
        total_losses = sum(h['failure'] for h in hourly.values())
        total_pnl = sum(h['net_pnl'] for h in hourly.values())
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        totals_row += f" {total_trades:^8} {win_rate:>5.1f}% ${total_pnl:>9,.0f}"
    print(totals_row)
    print("=" * 100)


# ============================================================================
# TOP-LEVEL ORCHESTRATOR
# ============================================================================

async def run_risk_gradient_analysis(args, logger=None):
    """
    Run risk gradient analysis using parsed args.

    This is the top-level orchestrator called from analyze_credit_spread_intervals.py
    --mode risk-gradient or from the thin wrapper script.

    Args:
        args: Parsed command-line arguments namespace.
        logger: Logger instance. If None, one is created.

    Returns:
        0 on success, 1 on failure.
    """
    # Lazy imports to avoid import errors when DB is not needed
    from common.questdb_db import StockQuestDB
    if logger is None:
        from common.logging_utils import get_logger
        log_level = getattr(args, 'log_level', 'INFO')
        logger = get_logger("ndx_risk_gradient", level=log_level)

    # Get database connection string
    db_path = getattr(args, 'db_path', None)
    if db_path is None:
        db_path = os.environ.get('QUEST_DB_STRING')

    if not db_path:
        logger.error("No database connection string provided. Set QUEST_DB_STRING or use --db-path")
        return 1

    # Resolve output directory
    output_dir_str = getattr(args, 'output_dir', None) or str(Path(__file__).resolve().parent.parent)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize ticker
    ticker = getattr(args, 'underlying_ticker', None) or getattr(args, 'ticker', 'NDX')
    ticker = ticker.replace('I:', '').upper()

    lookback_days_list = getattr(args, 'lookback_days', [90, 180])
    risk_cap = getattr(args, 'risk_cap', 500000)
    min_trading_hour = getattr(args, 'min_trading_hour', 9)
    max_trading_hour = getattr(args, 'max_trading_hour', 12)
    gradient_steps = getattr(args, 'gradient_steps', 7)
    step_size = getattr(args, 'step_size', 0.0025)
    processes = getattr(args, 'processes', 8)

    print("=" * 80)
    print("NDX RISK-BASED ANALYSIS")
    print("=" * 80)
    print(f"Ticker: {ticker}")
    print(f"Lookback periods: {lookback_days_list} days")
    print(f"Risk cap: ${risk_cap:,}")
    print(f"Trading hours: {min_trading_hour}:00 - {max_trading_hour}:00 PT")
    print(f"Gradient steps: {gradient_steps} (step size: {step_size * 100:.2f}%)")
    print("=" * 80)

    db = StockQuestDB(db_path, enable_cache=True, logger=logger)

    try:
        all_metrics = {}
        for lookback_days in lookback_days_list:
            print(f"\n[1] Querying {lookback_days}-day historical price movements...")
            metrics = await query_historical_max_movements(db, ticker, lookback_days, logger)
            if metrics:
                all_metrics[lookback_days] = metrics
            else:
                logger.warning(f"Skipping {lookback_days}-day analysis - no data")

        if not all_metrics:
            logger.error("No historical data found for any lookback period")
            return 1

        print_safe_points_comparison(all_metrics, logger)

        configs_generated = []

        for lookback_days, metrics in all_metrics.items():
            for metric_type in ['intraday', 'close_to_close']:
                safe_point_put = metrics[metric_type]['put']
                safe_point_call = metrics[metric_type]['call']

                gradient_values = generate_risk_gradient_values(
                    safe_point_put,
                    safe_point_call,
                    gradient_steps,
                    step_size
                )

                print_gradient_preview(
                    gradient_values,
                    lookback_days,
                    metric_type,
                    safe_point_put,
                    safe_point_call
                )

                config = create_grid_config(
                    gradient_values,
                    args,
                    lookback_days,
                    metric_type,
                    safe_point_put,
                    safe_point_call
                )

                metric_short = 'intra' if metric_type == 'intraday' else 'c2c'
                config_filename = f"grid_config_ndx_risk_gradient_{lookback_days}d_{metric_short}.json"
                config_path = output_dir / config_filename

                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                print(f"\n[2] Generated config: {config_path}")

                results_filename = f"ndx_risk_gradient_{lookback_days}d_{metric_short}_results.csv"
                results_path = output_dir / results_filename

                configs_generated.append({
                    'config_path': str(config_path),
                    'results_path': str(results_path),
                    'lookback_days': lookback_days,
                    'metric_type': metric_type,
                    'gradient_values': gradient_values
                })

        generate_config_only = getattr(args, 'generate_config_only', False)
        run_backtest = getattr(args, 'run_backtest', False)

        if generate_config_only:
            print("\n[!] Config generation complete. Use --run-backtest to execute analysis.")
            print("\nGenerated configs:")
            for cfg in configs_generated:
                print(f"  - {cfg['config_path']}")
            return 0

        if run_backtest:
            print("\n" + "=" * 80)
            print("RUNNING BACKTESTS")
            print("=" * 80)

            for cfg in configs_generated:
                print(f"\n[3] Running backtest for {cfg['lookback_days']}-day {cfg['metric_type']}...")
                result_path = await run_backtest_for_config(
                    cfg['config_path'],
                    cfg['results_path'],
                    processes,
                    logger
                )

                if result_path:
                    parse_and_display_results(
                        result_path,
                        cfg['gradient_values'],
                        cfg['lookback_days'],
                        cfg['metric_type'],
                        logger,
                        step_size
                    )
        else:
            print("\n[!] To run backtests, add --run-backtest flag")
            print("\nTo run manually:")
            for cfg in configs_generated:
                print(f"  python scripts/analyze_credit_spread_intervals.py \\")
                print(f"    --grid-config {cfg['config_path']} \\")
                print(f"    --grid-output {cfg['results_path']} \\")
                print(f"    --grid-sort win_rate --processes {processes}")

    finally:
        await db.close()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return 0
