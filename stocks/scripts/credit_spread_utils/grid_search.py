"""
Grid search optimization engine for credit spread parameters.

Supports parallel execution, resume capability, and configurable
parameter sweeps across all spread configuration dimensions.
"""

import asyncio
import csv
import itertools
import json
import multiprocessing
import os
import time
from typing import Dict, List, Optional, Any

from common.questdb_db import StockQuestDB
from common.logging_utils import get_logger

from .interval_analyzer import analyze_interval
from .spread_builder import parse_percent_beyond, parse_max_spread_width, parse_min_premium_diff
from .metrics import compute_metrics, filter_top_n_per_day
from .data_loader import find_csv_files_in_dir, load_data_cached
from .timezone_utils import resolve_timezone
from .delta_utils import DeltaFilterConfig, parse_delta_range
from .strategies import StrategyRegistry


def _build_strategy_for_combo(strategy_config: dict, combo: dict, logger=None):
    """Build a strategy instance for a grid combo, applying feature flag overrides.

    Extracts any 'strategy.feature_flags.*' keys from the combo and merges
    them into the strategy config before creating the strategy.

    Args:
        strategy_config: Base strategy config dict from grid JSON
        combo: Grid parameter combination
        logger: Optional logger

    Returns:
        Configured strategy instance, or None if no strategy configured
    """
    if not strategy_config:
        return None

    config_dict = dict(strategy_config)
    feature_flags = dict(config_dict.get('feature_flags', {}))

    # Extract strategy.feature_flags.* from combo
    for key, value in list(combo.items()):
        if key.startswith('strategy.feature_flags.'):
            flag_name = key[len('strategy.feature_flags.'):]
            feature_flags[flag_name] = value

    config_dict['feature_flags'] = feature_flags
    name = config_dict.get('name', config_dict.get('strategy', 'single_entry'))

    return StrategyRegistry.create(name, config_dict, logger=logger)


def _float_range(start, stop, step):
    """Generate float values from start to stop (inclusive) with given step."""
    values = []
    current = start
    while current <= stop + step * 0.01:
        values.append(round(current, 6))
        current += step
    return values


def _expand_grid_param(name, spec):
    """Expand a grid parameter specification into a list of values."""
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        if 'min' in spec and 'max' in spec and 'step' in spec:
            return _float_range(spec['min'], spec['max'], spec['step'])
        raise ValueError(f"Dict spec for '{name}' must have min, max, step keys")
    return [spec]


def _generate_combinations(grid_params: dict) -> List[dict]:
    """Generate all parameter combinations from the grid specification."""
    param_names = []
    param_values = []
    for name, spec in grid_params.items():
        param_names.append(name)
        param_values.append(_expand_grid_param(name, spec))

    combinations = []
    for values in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, values)))
    return combinations


def _combo_to_key(combo: dict) -> tuple:
    """Create a hashable key from a parameter combination."""
    # Convert dict/list values to JSON strings for hashability
    hashable_items = []
    for k, v in sorted(combo.items()):
        if isinstance(v, (dict, list)):
            hashable_items.append((k, json.dumps(v, sort_keys=True)))
        else:
            hashable_items.append((k, v))
    return tuple(hashable_items)


def _load_existing_grid_results(csv_path: str) -> set:
    """Load existing grid results to support resume."""
    existing_keys = set()
    if not os.path.exists(csv_path):
        return existing_keys
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        param_cols = [
            'option_type', 'percent_beyond_put', 'percent_beyond_call',
            'max_spread_width', 'max_spread_width_put', 'max_spread_width_call',
            'min_contract_price', 'max_credit_width_ratio',
            'max_strike_distance_pct', 'min_trading_hour', 'max_trading_hour',
            'profit_target_pct', 'min_premium_diff', 'min_premium_diff_put', 'min_premium_diff_call',
            'max_short_delta', 'min_short_delta', 'max_long_delta', 'min_long_delta',
            'delta_range', 'require_delta', 'delta_default_iv', 'use_vix1d',
        ]
        # Also include any strategy.feature_flags.* columns
        if reader.fieldnames:
            for col in reader.fieldnames:
                if col.startswith('strategy.feature_flags.') and col not in param_cols:
                    param_cols.append(col)
        for row in reader:
            combo = {}
            for col in param_cols:
                if col in row and row[col]:
                    try:
                        val = float(row[col])
                        if val == int(val) and '.' not in row[col]:
                            val = int(val)
                        combo[col] = val
                    except (ValueError, TypeError):
                        combo[col] = row[col]
            existing_keys.add(_combo_to_key(combo))
    return existing_keys


async def run_backtest_with_params(
    interval_groups,
    db,
    params: dict,
    logger,
    strategy=None,
) -> dict:
    """Run a full backtest with given params, return metrics dict.

    Args:
        interval_groups: Pre-grouped interval DataFrames
        db: Database connection
        params: Parameter dictionary
        logger: Logger instance
        strategy: Optional strategy instance for strategy-aware grid search

    Returns:
        Metrics dictionary with performance statistics
    """
    results = []
    option_types = ['call', 'put'] if params.get('option_type', 'both') == 'both' else [params['option_type']]

    # Construct percent_beyond tuple from separate put/call keys or parse combined format
    percent_beyond_raw = params.get('percent_beyond')
    if percent_beyond_raw is not None:
        if isinstance(percent_beyond_raw, str):
            # Parse string format (e.g., "0.02" or "0.02:0.03")
            try:
                percent_beyond = parse_percent_beyond(percent_beyond_raw)
            except ValueError:
                percent_beyond = (0.02, 0.02)
        elif isinstance(percent_beyond_raw, (int, float)):
            # Single numeric value - use for both
            percent_beyond = (float(percent_beyond_raw), float(percent_beyond_raw))
        elif isinstance(percent_beyond_raw, (list, tuple)) and len(percent_beyond_raw) == 2:
            percent_beyond = (float(percent_beyond_raw[0]), float(percent_beyond_raw[1]))
        else:
            percent_beyond = (0.02, 0.02)
    else:
        # Fallback to separate put/call keys
        percent_beyond = (params.get('percent_beyond_put', 0.02), params.get('percent_beyond_call', 0.02))

    # Construct max_spread_width tuple from separate put/call keys or parse combined format
    max_spread_width_raw = params.get('max_spread_width', 200)
    if isinstance(max_spread_width_raw, str):
        # Parse string format (e.g., "30" or "30:40")
        try:
            max_spread_width = parse_max_spread_width(max_spread_width_raw)
        except ValueError:
            max_spread_width = (200, 200)
    elif isinstance(max_spread_width_raw, (int, float)):
        max_spread_width = (float(max_spread_width_raw), float(max_spread_width_raw))
    elif isinstance(max_spread_width_raw, (list, tuple)) and len(max_spread_width_raw) == 2:
        max_spread_width = (float(max_spread_width_raw[0]), float(max_spread_width_raw[1]))
    else:
        max_spread_width = (
            params.get('max_spread_width_put', 200),
            params.get('max_spread_width_call', 200)
        )

    # Parse dynamic_spread_width config if provided
    dynamic_width_config = None
    dynamic_width_raw = params.get('dynamic_spread_width')
    if dynamic_width_raw:
        from credit_spread_utils.dynamic_width_utils import DynamicWidthConfig
        if isinstance(dynamic_width_raw, dict):
            dynamic_width_config = DynamicWidthConfig.from_dict(dynamic_width_raw)
        elif isinstance(dynamic_width_raw, str):
            from credit_spread_utils.dynamic_width_utils import parse_dynamic_width_config
            dynamic_width_config = parse_dynamic_width_config(dynamic_width_raw)

    # Build delta filter config if any delta params are provided
    delta_filter_config = None
    if any(params.get(k) is not None for k in ['max_short_delta', 'min_short_delta', 'max_long_delta',
                                                 'min_long_delta', 'delta_range']) or params.get('require_delta'):
        # Parse delta_range if provided (overrides min/max_short_delta)
        min_short_delta = params.get('min_short_delta')
        max_short_delta = params.get('max_short_delta')
        if params.get('delta_range'):
            parsed_min, parsed_max = parse_delta_range(params['delta_range'])
            if parsed_min is not None:
                min_short_delta = parsed_min
            if parsed_max is not None:
                max_short_delta = parsed_max

        delta_filter_config = DeltaFilterConfig(
            max_short_delta=max_short_delta,
            min_short_delta=min_short_delta,
            max_long_delta=params.get('max_long_delta'),
            min_long_delta=params.get('min_long_delta'),
            require_delta=params.get('require_delta', False),
            default_iv=params.get('delta_default_iv', 0.20),
            use_vix1d=params.get('use_vix1d', False),
            vix1d_dir=params.get('vix1d_dir', '../equities_output/I:VIX1D'),
        )

    for interval_time, interval_df in interval_groups:
        for opt_type in option_types:
            # Construct min_premium_diff tuple from separate put/call keys or fallback to single value
            min_premium_diff_default = params.get('min_premium_diff')
            min_premium_diff = None
            if min_premium_diff_default is not None:
                if isinstance(min_premium_diff_default, str):
                    # Parse if it's a string (put:call format)
                    try:
                        min_premium_diff = parse_min_premium_diff(min_premium_diff_default)
                    except ValueError:
                        logger.warning(f"Invalid min_premium_diff format: {min_premium_diff_default}")
                        min_premium_diff = None
                elif isinstance(min_premium_diff_default, (int, float)):
                    # Single value - use for both puts and calls
                    min_premium_diff = (float(min_premium_diff_default), float(min_premium_diff_default))
                else:
                    # Already a tuple or dict with separate keys
                    if isinstance(min_premium_diff_default, dict):
                        put_val = min_premium_diff_default.get('put', min_premium_diff_default.get('default'))
                        call_val = min_premium_diff_default.get('call', min_premium_diff_default.get('default'))
                        if put_val is not None and call_val is not None:
                            min_premium_diff = (float(put_val), float(call_val))
                    elif isinstance(min_premium_diff_default, (list, tuple)) and len(min_premium_diff_default) == 2:
                        min_premium_diff = (float(min_premium_diff_default[0]), float(min_premium_diff_default[1]))

            # Fallback to separate put/call keys if available
            if min_premium_diff is None:
                min_premium_diff_put = params.get('min_premium_diff_put')
                min_premium_diff_call = params.get('min_premium_diff_call')
                if min_premium_diff_put is not None and min_premium_diff_call is not None:
                    min_premium_diff = (float(min_premium_diff_put), float(min_premium_diff_call))
                elif min_premium_diff_put is not None:
                    min_premium_diff = (float(min_premium_diff_put), float(min_premium_diff_put))
                elif min_premium_diff_call is not None:
                    min_premium_diff = (float(min_premium_diff_call), float(min_premium_diff_call))

            result = await analyze_interval(
                db,
                interval_df,
                opt_type,
                percent_beyond,
                params.get('risk_cap'),
                params.get('min_spread_width', 5),
                max_spread_width,
                params.get('use_mid_price', False),
                params.get('min_contract_price', 0),
                params.get('underlying_ticker'),
                logger,
                params.get('max_credit_width_ratio', 0.60),
                params.get('max_strike_distance_pct'),
                False,  # use_current_price
                params.get('max_trading_hour', 15),
                params.get('min_trading_hour'),
                params.get('profit_target_pct'),
                params.get('output_tz'),
                params.get('force_close_hour'),
                min_premium_diff,
                dynamic_width_config,
                delta_filter_config,
            )
            if result:
                results.append(result)

    # Apply top_n filter if specified
    if params.get('top_n') and results:
        results = filter_top_n_per_day(results, params['top_n'])

    # If a strategy is provided, run it on the aggregated results
    if strategy and results:
        from collections import defaultdict
        from datetime import datetime as _dt
        import pandas as _pd

        results_by_date = defaultdict(lambda: {'put': [], 'call': []})
        for result in results:
            ts = result['timestamp']
            if hasattr(ts, 'date'):
                trading_date = ts.date()
            else:
                trading_date = _pd.to_datetime(ts).date()
            opt_type = result.get('option_type', 'unknown').lower()
            if opt_type in ['put', 'call']:
                results_by_date[trading_date][opt_type].append(result)

        strategy_results = []
        for trading_date, type_results in sorted(results_by_date.items()):
            for opt_type in ['put', 'call']:
                day_results = type_results[opt_type]
                if not day_results:
                    continue
                first_result = day_results[0]
                prev_close = first_result.get('prev_close')
                current_close = first_result.get('current_close')
                if prev_close is None or current_close is None:
                    continue

                trading_datetime = _dt.combine(trading_date, _dt.min.time())
                try:
                    positions = strategy.select_entries(
                        day_results=day_results,
                        prev_close=prev_close,
                        option_type=opt_type,
                        trading_date=trading_datetime,
                    )
                    strat_result = strategy.calculate_pnl(
                        positions=positions,
                        close_price=current_close,
                        option_type=opt_type,
                        trading_date=trading_datetime,
                    )
                    strategy_results.append(strat_result)
                except Exception:
                    continue

        # Build metrics from strategy results
        if strategy_results:
            total_trades = len(strategy_results)
            wins = sum(1 for sr in strategy_results if sr.total_pnl and sr.total_pnl > 0)
            total_credits = sum(sr.total_credit for sr in strategy_results)
            gains = sum(sr.total_pnl for sr in strategy_results if sr.total_pnl and sr.total_pnl > 0)
            losses = sum(abs(sr.total_pnl) for sr in strategy_results if sr.total_pnl and sr.total_pnl < 0)
            net_pnl = sum(sr.total_pnl or 0 for sr in strategy_results)
            total_max_loss = sum(sr.total_max_loss for sr in strategy_results)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            profit_factor = (gains / losses) if losses > 0 else float('inf')
            roi = (net_pnl / total_max_loss * 100) if total_max_loss > 0 else 0

            return {
                'total_trades': total_trades,
                'win_rate': round(win_rate, 1),
                'total_credits': round(total_credits, 2),
                'total_gains': round(gains, 2),
                'total_losses': round(losses, 2),
                'net_pnl': round(net_pnl, 2),
                'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
                'roi': round(roi, 2),
            }

    return compute_metrics(results)


def _format_grid_top_results(results: List[dict], sort_by: str, top_n: int) -> str:
    """Format top grid search results for terminal display."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f" TOP {min(top_n, len(results))} RESULTS (sorted by {sort_by})")
    lines.append(f"{'='*80}")

    for i, r in enumerate(results[:top_n], 1):
        combo = r['combo']
        m = r['metrics']

        pf_str = f"{m['profit_factor']:.1f}" if m['profit_factor'] != float('inf') else 'inf'
        pb_put = combo.get('percent_beyond_put', '-')
        pb_call = combo.get('percent_beyond_call', '-')

        # Format max_spread_width - show put:call if separate values, else single value
        msw_put = combo.get('max_spread_width_put', combo.get('max_spread_width', '-'))
        msw_call = combo.get('max_spread_width_call', combo.get('max_spread_width', '-'))
        if msw_put == msw_call:
            msw_str = str(msw_put)
        else:
            msw_str = f"{msw_put}:{msw_call}"

        param_str = (
            f"type={combo.get('option_type', '-')} "
            f"pb={pb_put}:{pb_call} "
            f"msw={msw_str} "
            f"mcp={combo.get('min_contract_price', '-')} "
            f"mcr={combo.get('max_credit_width_ratio', '-')} "
            f"msd={combo.get('max_strike_distance_pct', '-')} "
            f"mih={combo.get('min_trading_hour', '-')} "
            f"mth={combo.get('max_trading_hour', '-')} "
            f"ptp={combo.get('profit_target_pct', '-')}"
        )

        lines.append(
            f"#{i:<3} Net P&L: ${m['net_pnl']:>10,.2f}  "
            f"PF: {pf_str:<5}  "
            f"WR: {m['win_rate']:.0f}%  "
            f"Trades: {m['total_trades']:<4}  "
            f"| {param_str}"
        )
    return '\n'.join(lines)


def _write_grid_results_csv(results: List[dict], output_path: str):
    """Write grid search results to CSV."""
    if not results:
        return

    all_param_keys = set()
    for r in results:
        all_param_keys.update(r['combo'].keys())
    param_cols = sorted(all_param_keys)
    metric_cols = ['total_trades', 'win_rate', 'total_credits', 'total_gains',
                   'total_losses', 'net_pnl', 'profit_factor', 'roi']
    fieldnames = ['rank'] + param_cols + metric_cols

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(results, 1):
            row = {'rank': i}
            row.update(r['combo'])
            row.update(r['metrics'])
            writer.writerow(row)

    print(f"\nResults written to: {output_path} ({len(results)} rows)")


def _run_grid_combo_worker(args_tuple):
    """Worker function to run a single grid combination in a separate process."""
    combo, fixed_params, csv_paths, cache_dir, no_cache, log_level = args_tuple[:6]
    strategy_config = args_tuple[6] if len(args_tuple) > 6 else None

    logger = get_logger(f"grid_worker_{os.getpid()}", level=log_level)

    try:
        # Load data from cache (fast)
        df = load_data_cached(csv_paths, cache_dir=cache_dir, no_cache=no_cache, logger=None)

        # Create DB connection for this worker
        db_path = fixed_params.get('db_path')
        if isinstance(db_path, str) and db_path.startswith('$'):
            db_path = os.environ.get(db_path[1:], None)

        db = StockQuestDB(
            db_path,
            enable_cache=not fixed_params.get('no_cache', False),
            logger=None  # Reduce noise
        )

        # Group intervals
        interval_groups = list(df.groupby('interval'))

        # Resolve output timezone
        output_tz = None
        if fixed_params.get('output_timezone'):
            try:
                output_tz = resolve_timezone(fixed_params['output_timezone'])
            except Exception:
                pass

        # Build params
        params = dict(fixed_params)
        params.update(combo)
        params['output_tz'] = output_tz

        # Build strategy for this combo if configured
        strat = _build_strategy_for_combo(strategy_config, combo, logger=logger) if strategy_config else None

        # Run backtest
        metrics = asyncio.run(_run_backtest_with_params_sync(interval_groups, db, params, logger, strategy=strat))

        # Close DB
        asyncio.run(db.close())

        return {'combo': combo, 'metrics': metrics, 'success': True}

    except Exception as e:
        return {'combo': combo, 'error': str(e), 'success': False}


async def _run_backtest_with_params_sync(interval_groups, db, params, logger, strategy=None):
    """Wrapper for run_backtest_with_params for use in worker processes."""
    return await run_backtest_with_params(interval_groups, db, params, logger, strategy=strategy)


async def run_grid_search(args):
    """Run the grid search optimization mode."""
    logger = get_logger("grid_search", level=getattr(args, 'log_level', 'INFO'))

    # Load grid config
    with open(args.grid_config, 'r') as f:
        config = json.load(f)

    fixed_params = config.get('fixed_params', {})
    grid_params = config.get('grid_params', {})
    strategy_config = config.get('strategy', None)

    if not grid_params:
        print("Error: grid_params section is empty in config")
        return 1

    # Generate combinations
    combinations = _generate_combinations(grid_params)
    total_combos = len(combinations)

    # Show grid info
    print(f"Grid Search Configuration:")
    print(f"  Config file: {args.grid_config}")
    if strategy_config:
        strategy_name = strategy_config.get('name', strategy_config.get('strategy', 'unknown'))
        print(f"  Strategy: {strategy_name}")
        if strategy_config.get('config_file'):
            print(f"  Strategy config: {strategy_config['config_file']}")
        if strategy_config.get('feature_flags'):
            print(f"  Base feature flags: {strategy_config['feature_flags']}")
    print(f"  Parameters: {len(grid_params)}")
    for name, spec in grid_params.items():
        values = _expand_grid_param(name, spec)
        if len(values) <= 5:
            print(f"    {name}: {len(values)} values {values}")
        else:
            print(f"    {name}: {len(values)} values [{values[0]} ... {values[-1]}]")
    print(f"  Total combinations: {total_combos:,}")
    print(f"  Sort by: {args.grid_sort}")
    print(f"  Output: {args.grid_output}")

    if args.grid_dry_run:
        print(f"\n--grid-dry-run: Would run {total_combos:,} backtests. Exiting.")
        return 0

    # Resolve CSV paths from fixed_params
    csv_paths = []
    if 'csv_path' in fixed_params:
        import glob as glob_module
        raw_paths = fixed_params['csv_path']
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        for p in raw_paths:
            p = os.path.expandvars(p)
            expanded = glob_module.glob(p)
            if expanded:
                csv_paths.extend(sorted(expanded))
            else:
                csv_paths.append(p)
    elif 'csv_dir' in fixed_params and fixed_params.get('underlying_ticker'):
        ticker = fixed_params['underlying_ticker']
        found = find_csv_files_in_dir(
            fixed_params['csv_dir'], ticker,
            fixed_params.get('start_date'), fixed_params.get('end_date'),
            logger
        )
        csv_paths = [str(p) for p in found]

    if not csv_paths:
        print("Error: No CSV files resolved from fixed_params")
        return 1

    print(f"  CSV files: {len(csv_paths)}")

    # Load data (ONCE) using cache
    cache_dir = getattr(args, 'cache_dir', '.options_cache')
    no_cache = getattr(args, 'no_data_cache', False)
    try:
        df = load_data_cached(csv_paths, cache_dir=cache_dir, no_cache=no_cache, logger=logger)
    except (ValueError, Exception) as e:
        print(f"Error loading data: {e}")
        return 1

    print(f"  Loaded {len(df):,} rows, {df['interval'].nunique()} intervals")

    # Initialize DB (ONCE)
    db_path = fixed_params.get('db_path')
    if isinstance(db_path, str) and db_path.startswith('$'):
        db_path = os.environ.get(db_path[1:], None)
    db = StockQuestDB(
        db_path,
        enable_cache=not fixed_params.get('no_cache', False),
        logger=logger
    )

    # Pre-group intervals (ONCE)
    interval_groups = list(df.groupby('interval'))
    print(f"  Interval groups: {len(interval_groups)}")

    # Resolve output timezone
    output_tz = None
    if fixed_params.get('output_timezone'):
        try:
            output_tz = resolve_timezone(fixed_params['output_timezone'])
        except Exception:
            pass

    # Resume support
    existing_keys = set()
    if args.grid_resume:
        existing_keys = _load_existing_grid_results(args.grid_output)
        if existing_keys:
            print(f"  Resuming: {len(existing_keys)} existing results found, skipping those.")

    # Filter pending combos
    pending_combos = []
    for combo in combinations:
        if _combo_to_key(combo) not in existing_keys:
            pending_combos.append(combo)

    if not pending_combos:
        print("\nAll combinations already completed. Nothing to run.")
        await db.close()
        return 0

    # Determine number of processes
    num_processes = getattr(args, 'processes', 1) or 1
    if num_processes == 0:
        num_processes = multiprocessing.cpu_count()
    use_parallel = num_processes > 1

    print(f"  Combinations to run: {len(pending_combos):,}")
    print(f"  Parallel processes: {num_processes}")
    print(f"\nStarting grid search...")
    print("-" * 80)

    # Run grid search
    successful_results = []
    failed_count = 0
    start_time = time.time()

    if use_parallel:
        # Parallel execution using multiprocessing
        await db.close()  # Close DB - workers will create their own

        # Prepare worker args (strategy_config is JSON-serializable, no pickling issues)
        log_level = getattr(args, 'log_level', 'WARNING')
        worker_args = [
            (combo, fixed_params, csv_paths, cache_dir, no_cache, log_level, strategy_config)
            for combo in pending_combos
        ]

        completed = 0
        with multiprocessing.Pool(processes=num_processes) as pool:
            for result in pool.imap_unordered(_run_grid_combo_worker, worker_args):
                completed += 1
                if result['success']:
                    metrics = result['metrics']
                    if metrics['total_trades'] > 0:
                        successful_results.append({'combo': result['combo'], 'metrics': metrics})
                        print(
                            f"  [{completed}/{len(pending_combos)}] OK  "
                            f"Net P&L: ${metrics['net_pnl']:>10,.2f}  "
                            f"PF: {metrics['profit_factor']:.2f}  "
                            f"WR: {metrics['win_rate']:.0f}%  "
                            f"Trades: {metrics['total_trades']}"
                        )
                    else:
                        failed_count += 1
                else:
                    failed_count += 1
                    if completed <= 5 or completed % 100 == 0:
                        print(f"  [{completed}/{len(pending_combos)}] FAIL: {result.get('error', 'Unknown')[:80]}")
    else:
        # Sequential execution
        try:
            for idx, combo in enumerate(pending_combos, 1):
                # Build params from combo + fixed_params
                params = dict(fixed_params)
                params.update(combo)
                params['output_tz'] = output_tz

                # Build strategy for this combo if configured
                strat = _build_strategy_for_combo(strategy_config, combo, logger=logger) if strategy_config else None

                try:
                    metrics = await run_backtest_with_params(interval_groups, db, params, logger, strategy=strat)
                    if metrics['total_trades'] > 0:
                        successful_results.append({'combo': combo, 'metrics': metrics})
                        print(
                            f"  [{idx}/{len(pending_combos)}] OK  "
                            f"Net P&L: ${metrics['net_pnl']:>10,.2f}  "
                            f"PF: {metrics['profit_factor']:.2f}  "
                            f"WR: {metrics['win_rate']:.0f}%  "
                            f"Trades: {metrics['total_trades']}"
                        )
                    else:
                        failed_count += 1
                        if idx <= 5 or idx % 100 == 0:
                            print(f"  [{idx}/{len(pending_combos)}] SKIP: No trades")
                except Exception as e:
                    failed_count += 1
                    if idx <= 5 or idx % 100 == 0:
                        print(f"  [{idx}/{len(pending_combos)}] FAIL: {str(e)[:80]}")

        finally:
            await db.close()

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Completed in {elapsed:.1f}s | Successful: {len(successful_results)} | Failed/Skipped: {failed_count}")

    if not successful_results:
        print("\nNo successful results to report.")
        return 0

    # Sort results
    def sort_key(r):
        val = r['metrics'].get(args.grid_sort, 0)
        if val == float('inf'):
            return float('inf')
        return val

    successful_results.sort(key=sort_key, reverse=True)

    # Display top results
    print(_format_grid_top_results(successful_results, args.grid_sort, args.grid_top_n))

    # Write CSV
    _write_grid_results_csv(successful_results, args.grid_output)

    return 0
